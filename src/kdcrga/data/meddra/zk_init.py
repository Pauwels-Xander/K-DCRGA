"""MedDRA ancestor-average z_k initialisation for held-out side effects (RQ5).

For each held-out SE, average the TRAINED z_k of train SEs sharing the nearest
MedDRA ancestor (same HLT -> else HLGT -> else SOC). Fall back to the global mean
of train z_k when no shared ancestor exists or the held-out SE is unmatched.
"""
from __future__ import annotations

import torch


def meddra_zk_init(trained_z_k: torch.Tensor, train_ks: list[int],
                   held_out_ks: list[int], mapping) -> torch.Tensor:
    """Return Tensor[len(held_out_ks), dim]. `mapping`: {se_index: MeddraEntry}."""
    dim = trained_z_k.size(1)
    if train_ks:
        global_mean = trained_z_k[torch.tensor(train_ks)].mean(0)
    else:
        global_mean = torch.zeros(dim)

    out = torch.empty(len(held_out_ks), dim)
    for i, h in enumerate(held_out_ks):
        hm = mapping.get(h)
        chosen = None
        if hm is not None:
            for level in ("hlt", "hlgt", "soc"):
                sibs = [k for k in train_ks
                        if k in mapping
                        and getattr(mapping[k], level) == getattr(hm, level)]
                if sibs:
                    chosen = trained_z_k[torch.tensor(sibs)].mean(0)
                    break
        out[i] = chosen if chosen is not None else global_mean
    return out
