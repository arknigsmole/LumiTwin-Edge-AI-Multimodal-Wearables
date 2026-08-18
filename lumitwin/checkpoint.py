import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_atomic(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, seed: int, best: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "seed": seed, "best": best, "rng": rng_state()}
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[int, int, float]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    restore_rng_state(payload["rng"])
    return int(payload["epoch"]), int(payload["seed"]), float(payload["best"])
