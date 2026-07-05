"""GPU pre-flight: confirm each model kind fits in VRAM and estimate runtime.

The --smoke sweeps force CPU, so peak GPU memory is otherwise untested before a
multi-hour run. This runs REAL training batches per model on cuda at the config's
batch size, reports peak VRAM (catching OOM gracefully), and estimates steady-
state per-batch time via a DIFFERENTIAL measurement: time N_WARM batches and
N_WARM+N_STEP batches, then per_batch = (t_long - t_short) / N_STEP. Subtracting
cancels the fixed one-time cost (CUDA init, kernel compile, graph->GPU transfer,
the single validation pass) that otherwise dominates a naive dt/n_batches.

Run: python experiments/preflight_gpu.py
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

from _sweep_common import DATA, build_model, load_data  # noqa: E402
from kdcrga.config import load_config  # noqa: E402
from kdcrga.training.loop import train  # noqa: E402

CONFIGS = [
    ("dc_rgcn_v1", "configs/dc_rgcn_v1.yaml", "dc_rgcn"),
    ("decagon_v1", "configs/decagon_v1.yaml", "decagon"),
    ("lagat_v1", "configs/lagat_v1.yaml", "lagat"),
    ("kdcrga_v4", "configs/kdcrga_v4.yaml", "kdcrga"),     # RQ3 no-ontology arm (plain graph)
    ("kdcrga_v5", "configs/kdcrga_v5.yaml", "kdcrga"),     # headline: z_k + ATC-injected graph
]
N_WARM = 5          # batches in the short timing run (large enough for allocator steady state)
N_STEP = 10         # extra batches in the long run; per-batch = (t_long-t_short)/N_STEP


def _time_train(model, data, t, bs, max_batches):
    t0 = time.time()
    train(model, data, lr=t["lr"], weight_decay=t["weight_decay"],
          gamma=t["class_weight_gamma"], batch_size=bs,
          max_epochs=1, patience=99, seed=0, device="cuda",
          max_batches=max_batches, max_val_pairs=256, show_progress=False)
    torch.cuda.synchronize()
    return time.time() - t0


def main() -> None:
    if not torch.cuda.is_available():
        print("No CUDA available - pre-flight is a no-op on this machine.")
        return
    if not DATA.exists():
        print(f"SKIP: {DATA} not found.")
        return

    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)}  ({total_vram:.1f} GB)\n")

    # Load each graph variant once, keyed by use_ontology_atc (plain vs ATC-
    # injected) — same as run_sweep. v5 MUST be profiled on the injected graph or
    # its ~496 extra nodes + rel_atc relation go unmeasured, defeating the point.
    plain = load_data(seed=0)
    graph_cache: dict[bool, object] = {False: plain}

    def _graph_for(cfg):
        use_atc = bool(cfg["model"].get("use_ontology_atc", False))
        if use_atc not in graph_cache:
            graph_cache[use_atc] = load_data(seed=0, cfg=cfg)
        return graph_cache[use_atc]

    n_train = int((plain.ddi.split == 0).sum())  # DDI split identical across variants
    print(f"Train DDI pairs (fixed split): {n_train:,}\n")
    print(f"{'config':<14}{'bs':>6}{'peak VRAM':>12}{'per-batch':>12}"
          f"{'batch/ep':>10}{'~epoch':>10}  status")
    print("-" * 76)

    BS_SWEEP = {"kdcrga_v4": [256, 128, 64], "kdcrga_v5": [256, 128, 64]}
    for name, path, kind in CONFIGS:
        cfg = load_config(path)
        data = _graph_for(cfg)
        t = cfg["training"]
        for bs in BS_SWEEP.get(name, [t["batch_size"]]):
            n_batches = -(-n_train // bs)  # ceil
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            try:
                torch.manual_seed(0)
                m_short = build_model(kind, cfg, data)
                t_short = _time_train(m_short, data, t, bs, N_WARM)
                del m_short
                gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
                torch.manual_seed(0)
                m_long = build_model(kind, cfg, data)
                t_long = _time_train(m_long, data, t, bs, N_WARM + N_STEP)
                del m_long
                peak = torch.cuda.max_memory_allocated() / 1e9
                per_batch = max(t_long - t_short, 0.0) / N_STEP   # steady state
                epoch_min = per_batch * n_batches / 60.0
                label = f"{name}@{bs}"
                print(f"{label:<14}{bs:>6}{peak:>10.2f}GB{per_batch:>10.2f}s"
                      f"{n_batches:>10}{epoch_min:>8.1f}m  OK")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    peak = torch.cuda.max_memory_allocated() / 1e9
                    print(f"{name}@{bs:<8}{bs:>6}{peak:>10.2f}GB{'':>32}"
                          f"  *** OOM *** (lower batch_size)")
                else:
                    raise
            finally:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    print("\nper-batch is steady-state (warmup/val overhead subtracted out).")
    print("~epoch = per-batch x batches/epoch (training only; add a small val pass).")
    print("Full sweep ~= sum(epoch x epochs-to-converge) x 3 seeds. kdcrga_v4 is")
    print("shared by RQ1 and RQ3 -- the resume/skip logic trains it once, not twice.")


if __name__ == "__main__":
    main()
