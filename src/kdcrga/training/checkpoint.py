"""Checkpoint save/load for resumable training runs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: Path | str,
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val: float,
    waited: int,
    history: dict[str, list[float]],
    seed: int,
    best_state: dict[str, torch.Tensor] | None = None,
) -> None:
    """Persist model, optimiser, and training metadata for resume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val": best_val,
        "waited": waited,
        "history": history,
        "seed": seed,
        "best_state": best_state,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path | str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Restore model (and optionally optimiser) from a checkpoint file."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def save_history(path: Path | str, history: dict[str, list[float]]) -> None:
    """Write epoch metrics as JSON for inspection without loading weights."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
