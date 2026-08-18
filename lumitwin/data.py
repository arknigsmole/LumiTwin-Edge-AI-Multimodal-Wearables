from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Record:
    subject: str
    site: str
    colour: np.ndarray
    sensor: np.ndarray
    lux: np.ndarray
    covariates: np.ndarray
    sleep: int | None
    phase: float | None
    exposure: np.ndarray | None
    onset: float | None
    spherical_equivalent: float | None


def subject_statistics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmean(values, axis=0)
    scale = np.nanstd(values, axis=0)
    return center, np.maximum(scale, 1e-6)


def normalize_subject(values: np.ndarray) -> np.ndarray:
    center, scale = subject_statistics(values)
    return (np.nan_to_num(values, nan=center) - center) / scale


def aggregate_colour(colour: np.ndarray, source_seconds: float = 30.0, target_minutes: float = 10.0) -> np.ndarray:
    stride = round(target_minutes * 60.0 / source_seconds)
    usable = colour.shape[0] // stride * stride
    if usable == 0:
        raise ValueError("record is shorter than one timeline bin")
    reshaped = colour[:usable].reshape(-1, stride, colour.shape[-1])
    return np.exp(np.log(np.maximum(reshaped, 1e-6)).mean(axis=1))


def causal_window(values: np.ndarray, end: int, length: int) -> np.ndarray:
    start = end - length + 1
    if start >= 0:
        return values[start : end + 1]
    padding = np.repeat(values[:1], -start, axis=0)
    return np.concatenate((padding, values[: end + 1]), axis=0)


def validate_manifest(frame: pd.DataFrame) -> None:
    required = {"subject", "site", "colour_path", "sensor_path", "lux_path"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"manifest misses fields: {sorted(missing)}")
    if frame["subject"].isna().any() or frame["site"].isna().any():
        raise ValueError("subject and site identifiers cannot be empty")


def subject_split(subjects: list[str], sites: list[str], seed: int, fractions: tuple[float, float, float] = (0.7, 0.15, 0.15)) -> dict[str, str]:
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("split fractions must sum to one")
    generator = np.random.default_rng(seed)
    assignments: dict[str, str] = {}
    for site in sorted(set(sites)):
        members = sorted({subject for subject, current in zip(subjects, sites, strict=True) if current == site})
        generator.shuffle(members)
        first = round(len(members) * fractions[0])
        second = first + round(len(members) * fractions[1])
        for subject in members[:first]:
            assignments[subject] = "train"
        for subject in members[first:second]:
            assignments[subject] = "validation"
        for subject in members[second:]:
            assignments[subject] = "test"
    return assignments


class MulticohortDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, manifest: Path, root: Path, split: str, seed: int = 17, timeline_bins: int = 1008) -> None:
        frame = pd.read_csv(manifest)
        validate_manifest(frame)
        assignments = subject_split(frame["subject"].astype(str).tolist(), frame["site"].astype(str).tolist(), seed)
        self.frame = frame[frame["subject"].astype(str).map(assignments) == split].reset_index(drop=True)
        self.root = root
        self.timeline_bins = timeline_bins

    def __len__(self) -> int:
        return len(self.frame)

    def _array(self, relative: str) -> np.ndarray:
        path = self.root / relative
        array = np.load(path, allow_pickle=False)
        if not np.isfinite(array).all():
            raise ValueError(f"non-finite samples in {relative}")
        return array.astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        row = self.frame.iloc[index]
        colour = self._array(str(row["colour_path"]))
        sensor = normalize_subject(self._array(str(row["sensor_path"]))).astype(np.float32)
        lux = self._array(str(row["lux_path"])).reshape(-1)
        alpha_path = row.get("alpha_path")
        if isinstance(alpha_path, str):
            alpha = self._array(alpha_path)
        else:
            alpha = np.pad(np.log1p(aggregate_colour(colour)), ((0, 0), (0, 2)))
        end = alpha.shape[0] - 1
        timeline = causal_window(alpha, end, self.timeline_bins)
        sensor_epoch = sensor[-min(400, sensor.shape[0]) :]
        if sensor_epoch.shape[0] < 400:
            sensor_epoch = np.pad(sensor_epoch, ((400 - sensor_epoch.shape[0], 0), (0, 0)), mode="edge")
        covariates = np.asarray([row.get(f"covariate_{position}", 0.0) for position in range(8)], dtype=np.float32)
        output = {"timeline": torch.from_numpy(timeline.astype(np.float32)), "sensor": torch.from_numpy(sensor_epoch), "lux": torch.from_numpy(causal_window(lux[:, None], len(lux) - 1, self.timeline_bins)[:, 0].astype(np.float32)), "covariates": torch.from_numpy(covariates), "sleep": torch.tensor(int(row.get("sleep", -1))), "phase": torch.tensor(float(row.get("phase", float("nan"))), dtype=torch.float32), "exposure": torch.tensor([float(row.get(f"exposure_{position}", float("nan"))) for position in range(4)]), "onset": torch.tensor(float(row.get("onset", float("nan"))), dtype=torch.float32), "spherical": torch.tensor(float(row.get("spherical_equivalent", float("nan"))), dtype=torch.float32)}
        return output


def collate_disjoint(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    keys = batch[0].keys()
    return {key: torch.stack([item[key] for item in batch]) for key in keys}
