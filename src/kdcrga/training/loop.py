"""Training loop: epoch iteration, validation, early stopping.

Mixed-batch triple training (encoder runs once per batch; the side effect enters
only via the decoder, so per-relation batching is not needed for this baseline).
Monitors validation AUPRC with early stopping (proposal Sec. 4.9).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from tqdm import tqdm

from kdcrga.eval.metrics import evaluate
from kdcrga.training.checkpoint import load_checkpoint, save_checkpoint, save_history
from kdcrga.training.loss import class_weights, weighted_bce_loss
from kdcrga.training.negatives import sample_negatives


def _log(msg: str) -> None:
    print(msg, flush=True)


def _progress_log(msg: str, *, enabled: bool) -> None:
    """Print without corrupting an active tqdm bar."""
    if enabled:
        tqdm.write(msg)
    else:
        _log(msg)


def _split_triples(data, split_id):
    m = data.ddi.split == split_id
    return data.ddi.pair_index[:, m], data.ddi.side_effect[m]


def train(
    model,
    data,
    lr: float = 5e-4,
    weight_decay: float = 1e-5,
    gamma: float = 0.5,
    batch_size: int = 1024,
    accum_steps: int = 1,
    max_epochs: int = 200,
    patience: int = 15,
    seed: int = 0,
    device: str = "cpu",
    max_batches: int | None = None,
    max_val_pairs: int | None = None,
    checkpoint_dir: Path | str | None = None,
    resume: bool = False,
    show_progress: bool = True,
) -> dict:
    """Train a decoder-conditioned model with mixed-batch triple sampling and
    early stopping on validation AUPRC. Returns a history dict.

    `accum_steps` accumulates gradients over that many micro-batches before each
    optimizer step, giving an effective batch of ``batch_size * accum_steps`` (used
    to match a common effective batch across VRAM-bound models); 1 = step per batch.
    `max_batches` caps batches per epoch (for fast sanity tests); None = all.
    `max_val_pairs` caps the per-epoch validation positives for fast sanity runs
    (negatives are 1:1, so val cost stays bounded); None = full val set.
    When `checkpoint_dir` is set, writes ``last.pt``, ``best.pt``, and
    ``history.json`` after every epoch. Pass ``resume=True`` to continue from
    ``last.pt`` in that directory.
    Side effect: sets ``torch.set_float32_matmul_precision('high')`` (enables TF32).
    """
    # TF32 on Ada: faster fp32 matmuls (R-GCN convs + attention) at negligible
    # accuracy cost. Idempotent global setting; harmless on CPU.
    torch.set_float32_matmul_precision("high")
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    g = torch.Generator().manual_seed(seed)
    num_drugs = int(data.num_drugs)
    num_se = int(data.num_side_effects)
    dev = torch.device(device)
    model.to(dev)
    data["entity"].x = data["entity"].x.to(dev)
    for et in list(data.edge_types):
        if et[1].startswith("rel_"):
            data[et].edge_index = data[et].edge_index.to(dev)
    data.ddi.pair_index = data.ddi.pair_index.to(dev)
    data.ddi.side_effect = data.ddi.side_effect.to(dev)
    if data.ddi.split is not None:
        data.ddi.split = data.ddi.split.to(dev)
    train_pairs, train_se = _split_triples(data, 0)
    val_pairs, val_se = _split_triples(data, 1)
    n = train_pairs.size(1)

    if show_progress:
        n_batches = len(range(0, n, batch_size))
        if max_batches is not None:
            n_batches = min(n_batches, max_batches)
        _log(
            f"Training on {device}: {n:,} train triples, "
            f"{n_batches} batches/epoch, max_epochs={max_epochs}"
        )
        _log("Building positive-triple index for negative sampling…")

    positives = {
        (int(data.ddi.pair_index[0, i]), int(data.ddi.pair_index[1, i]),
         int(data.ddi.side_effect[i]))
        for i in range(data.ddi.side_effect.numel())
    }

    if show_progress:
        _log(f"Index ready ({len(positives):,} positives). Starting epochs.")
    weights = class_weights(train_se, num_se, gamma=gamma)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: dict[str, list[float]] = {"train_loss": [], "val_auprc": []}
    best_val, best_state, waited = float("-inf"), None, 0
    start_epoch = 0

    last_ckpt = ckpt_dir / "last.pt" if ckpt_dir is not None else None
    if resume and last_ckpt is not None and last_ckpt.exists():
        payload = load_checkpoint(last_ckpt, model, opt)
        history = payload["history"]
        best_val = payload["best_val"]
        waited = payload["waited"]
        start_epoch = int(payload["epoch"]) + 1
        best_state = payload.get("best_state")
        if show_progress:
            _log(
                f"Resumed from {last_ckpt} at epoch {start_epoch}/{max_epochs} "
                f"(best val_auprc={best_val:.4f})"
            )

    batch_starts = list(range(0, n, batch_size))
    if max_batches is not None:
        batch_starts = batch_starts[:max_batches]

    for epoch in range(start_epoch, max_epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        epoch_loss = 0.0
        n_batches = 0
        batch_iter = tqdm(
            batch_starts,
            desc=f"Epoch {epoch + 1}/{max_epochs}",
            disable=not show_progress,
            file=sys.stdout,
            dynamic_ncols=True,
            mininterval=1.0,
            leave=False,
        )
        opt.zero_grad()
        pending = 0
        for start in batch_iter:
            idx = perm[start:start + batch_size]
            pos_pair, pos_se = train_pairs[:, idx], train_se[idx]
            neg_pair, neg_se = sample_negatives(
                pos_pair, pos_se, num_drugs, positives, seed=int(perm[start]))
            pair = torch.cat([pos_pair, neg_pair], dim=1)
            se = torch.cat([pos_se, neg_se], dim=0)
            labels = torch.cat([torch.ones(pos_se.numel()),
                                torch.zeros(neg_se.numel())]).to(dev)
            logits = model(data, pair, se)
            loss = weighted_bce_loss(logits, labels, se, weights)
            # Scale by 1/accum_steps so the accumulated gradient equals the mean
            # over the effective (batch_size * accum_steps) window.
            (loss / accum_steps).backward()
            pending += 1
            if pending == accum_steps:
                opt.step()
                opt.zero_grad()
                pending = 0
            epoch_loss += float(loss)
            n_batches += 1
            if show_progress:
                batch_iter.set_postfix(loss=f"{float(loss):.4f}", refresh=False)
        if pending > 0:        # flush a partial final accumulation window
            opt.step()
            opt.zero_grad()

        if show_progress:
            batch_iter.close()

        mean_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(mean_loss)

        model.eval()
        with torch.no_grad():
            v_pairs = val_pairs if max_val_pairs is None else val_pairs[:, :max_val_pairs]
            v_se = val_se if max_val_pairs is None else val_se[:max_val_pairs]
            vneg_pair, vneg_se = sample_negatives(
                v_pairs, v_se, num_drugs, positives, seed=seed)
            vpair = torch.cat([v_pairs, vneg_pair], dim=1)
            vse = torch.cat([v_se, vneg_se], dim=0)
            vlabels = torch.cat([torch.ones(v_se.numel()),
                                 torch.zeros(vneg_se.numel())]).to(dev)
            # Batch the val forward — K-DCRGA's neighbour-materialising readouts
            # OOM on an unbatched ~75k-pair forward (Plan 4 final-review finding).
            # Encode the static graph ONCE; the encoder is identical across chunks
            # in eval mode (weights frozen, no dropout).
            v_total = vpair.size(1)
            h_val = model.encode(data)
            v_logit_chunks: list[torch.Tensor] = []
            for vs in range(0, v_total, batch_size):
                v_logit_chunks.append(
                    model(data, vpair[:, vs:vs + batch_size], vse[vs:vs + batch_size], h=h_val)
                )
            vlogits = torch.cat(v_logit_chunks) if v_logit_chunks else torch.empty(0, device=dev)
            res = evaluate(vlabels, torch.sigmoid(vlogits), vse,
                           getattr(data, "quartiles", {}))
        val_auprc = res["auprc"]
        history["val_auprc"].append(val_auprc)

        if val_auprc > best_val:
            best_val, waited = val_auprc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1

        if show_progress:
            _progress_log(
                f"Epoch {epoch + 1}/{max_epochs} done — "
                f"train_loss={mean_loss:.4f} val_auprc={val_auprc:.4f} "
                f"best={best_val:.4f} patience={waited}/{patience}",
                enabled=True,
            )

        if ckpt_dir is not None:
            save_checkpoint(
                ckpt_dir / "last.pt",
                epoch=epoch,
                model=model,
                optimizer=opt,
                best_val=best_val,
                waited=waited,
                history=history,
                seed=seed,
                best_state=best_state,
            )
            save_history(ckpt_dir / "history.json", history)
            if waited == 0:
                save_checkpoint(
                    ckpt_dir / "best.pt",
                    epoch=epoch,
                    model=model,
                    optimizer=opt,
                    best_val=best_val,
                    waited=waited,
                    history=history,
                    seed=seed,
                    best_state=best_state,
                )

        if waited >= patience:
            if show_progress:
                _progress_log(f"Early stopping at epoch {epoch + 1}.", enabled=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history
