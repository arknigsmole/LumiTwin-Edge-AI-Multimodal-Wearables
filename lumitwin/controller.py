from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class OperatingPoint:
    bits: int
    duty: float
    energy: float
    distortion: float


class AQDCController:
    def __init__(self, energy_budget: float, penalty: float = 2.0) -> None:
        self.energy_budget = float(energy_budget)
        self.penalty = float(penalty)
        self.queue = 0.0

    def candidates(self, entropy: float, log_variance: float) -> tuple[OperatingPoint, ...]:
        points = []
        for bits in (4, 8):
            for duty in (1.0, 0.5, 0.25):
                energy = 0.22 + 0.48 * duty + 0.34 * bits / 8.0
                quantization = (8.0 / bits - 1.0) * (0.15 + entropy)
                sampling = (1.0 / duty - 1.0) * (0.08 + log_variance)
                points.append(OperatingPoint(bits, duty, energy, quantization + sampling))
        return tuple(points)

    def choose(self, entropy: float, log_variance: float) -> OperatingPoint:
        points = self.candidates(entropy, log_variance)
        chosen = min(points, key=lambda point: self.queue * point.energy + self.penalty * point.distortion)
        self.queue = max(self.queue + chosen.energy - self.energy_budget, 0.0)
        return chosen

    def state_dict(self) -> dict[str, float]:
        return {"energy_budget": self.energy_budget, "penalty": self.penalty, "queue": self.queue}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.energy_budget = float(state["energy_budget"])
        self.penalty = float(state["penalty"])
        self.queue = float(state["queue"])


def posterior_entropy(weights: Tensor) -> Tensor:
    probabilities = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)


def fake_quantize(tensor: Tensor, bits: int) -> Tensor:
    limit = 2 ** (bits - 1) - 1
    scale = tensor.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / limit
    quantized = torch.round(tensor / scale).clamp(-limit - 1, limit)
    return tensor + (quantized * scale - tensor).detach()


def duty_mask(length: int, duty: float, device: torch.device) -> Tensor:
    stride = max(1, round(1.0 / duty))
    indices = torch.arange(length, device=device)
    return indices.remainder(stride).eq(0)
