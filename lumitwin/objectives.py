from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .model import LumiTwinOutput
from .pacemaker import circular_error


@dataclass
class LossTerms:
    total: Tensor
    sleep: Tensor
    phase: Tensor
    exposure: Tensor
    onset: Tensor
    spherical: Tensor
    consistency: Tensor


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    selected = values[mask]
    if selected.numel() == 0:
        return values.sum() * 0.0
    return selected.mean()


class DisjointLabelObjective(nn.Module):
    def __init__(self, consistency_weight: float = 0.1) -> None:
        super().__init__()
        self.consistency_weight = consistency_weight

    def forward(self, output: LumiTwinOutput, batch: dict[str, Tensor]) -> LossTerms:
        sleep_mask = batch["sleep"] >= 0
        sleep_losses = torch.nn.functional.cross_entropy(output.sleep_logits, batch["sleep"].clamp_min(0), reduction="none")
        sleep = masked_mean(sleep_losses, sleep_mask)
        phase_mask = torch.isfinite(batch["phase"])
        phase = masked_mean(circular_error(output.phase_location, batch["phase"]), phase_mask)
        exposure_mask = torch.isfinite(batch["exposure"]).all(dim=-1)
        exposure_losses = torch.nn.functional.smooth_l1_loss(output.exposure, torch.nan_to_num(batch["exposure"]), reduction="none").mean(dim=-1)
        exposure = masked_mean(exposure_losses, exposure_mask)
        onset_mask = torch.isfinite(batch["onset"])
        onset_losses = torch.nn.functional.binary_cross_entropy_with_logits(output.onset_logit, torch.nan_to_num(batch["onset"]), reduction="none")
        onset = masked_mean(onset_losses, onset_mask)
        spherical_mask = torch.isfinite(batch["spherical"])
        spherical_losses = torch.nn.functional.smooth_l1_loss(output.spherical_equivalent, torch.nan_to_num(batch["spherical"]), reduction="none")
        spherical = masked_mean(spherical_losses, spherical_mask)
        melanopic_summary = batch["timeline"][..., 4].mean(dim=-1)
        representation_light = output.representation.square().mean(dim=-1).sqrt()
        consistency = torch.nn.functional.smooth_l1_loss(representation_light, torch.log1p(melanopic_summary.clamp_min(0.0)))
        total = sleep + phase + exposure + onset + spherical + self.consistency_weight * consistency
        return LossTerms(total, sleep, phase, exposure, onset, spherical, consistency)


def cohen_kappa(prediction: Tensor, target: Tensor, classes: int = 5) -> Tensor:
    matrix = torch.zeros(classes, classes, dtype=torch.float64, device=prediction.device)
    indices = target.long() * classes + prediction.long()
    matrix.flatten().scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.float64))
    total = matrix.sum().clamp_min(1.0)
    observed = matrix.diag().sum() / total
    expected = (matrix.sum(0) * matrix.sum(1)).sum() / total.square()
    return (observed - expected) / (1.0 - expected).clamp_min(1e-12)


def binary_auc(scores: Tensor, target: Tensor) -> Tensor:
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, scores.numel() + 1, device=scores.device, dtype=torch.float64)
    positive = target.bool()
    positive_count = positive.sum()
    negative_count = target.numel() - positive_count
    numerator = ranks[positive].sum() - positive_count * (positive_count + 1) / 2
    return numerator / (positive_count * negative_count).clamp_min(1)
