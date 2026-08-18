from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import nnls
from torch import Tensor, nn


@dataclass(frozen=True)
class SpectralGrid:
    wavelengths: np.ndarray
    responsivity: np.ndarray
    action_spectra: np.ndarray

    def validate(self) -> None:
        if self.wavelengths.ndim != 1:
            raise ValueError("wavelengths must be one-dimensional")
        if self.responsivity.shape != (3, self.wavelengths.size):
            raise ValueError("responsivity must have shape [3, wavelength]")
        if self.action_spectra.shape != (5, self.wavelengths.size):
            raise ValueError("action spectra must have shape [5, wavelength]")
        if np.any(np.diff(self.wavelengths) <= 0):
            raise ValueError("wavelengths must increase")


def gaussian_spectrum(wavelengths: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((wavelengths - center) / width) ** 2)


def planck_spectrum(wavelengths: np.ndarray, temperature: float) -> np.ndarray:
    meters = wavelengths * 1e-9
    c1 = 3.741771852e-16
    c2 = 1.438776877e-2
    density = c1 / (meters**5 * np.expm1(c2 / (meters * temperature)))
    return density / np.trapz(density, wavelengths)


def daylight_spectrum(wavelengths: np.ndarray, elevation: float) -> np.ndarray:
    blue = gaussian_spectrum(wavelengths, 460.0, 72.0)
    green = gaussian_spectrum(wavelengths, 555.0, 105.0)
    red = gaussian_spectrum(wavelengths, 620.0, 130.0)
    mix = np.clip(np.sin(np.deg2rad(elevation)), 0.05, 1.0)
    density = mix * blue + green + (1.0 - mix) * red
    return density / np.trapz(density, wavelengths)


def fluorescent_spectrum(wavelengths: np.ndarray, variant: int) -> np.ndarray:
    centers = ((405.0, 436.0, 546.0, 611.0), (420.0, 490.0, 545.0, 610.0), (435.0, 545.0, 580.0, 615.0), (405.0, 490.0, 555.0, 625.0))
    weights = ((0.3, 0.8, 1.0, 0.5), (0.4, 0.7, 1.0, 0.6), (0.7, 0.8, 0.9, 0.5), (0.5, 0.7, 1.0, 0.8))
    density = np.zeros_like(wavelengths, dtype=np.float64)
    for center, weight in zip(centers[variant], weights[variant], strict=True):
        density += weight * gaussian_spectrum(wavelengths, center, 8.0)
    return density / np.trapz(density, wavelengths)


def led_spectrum(wavelengths: np.ndarray, variant: int) -> np.ndarray:
    blue_centers = (445.0, 455.0, 465.0)
    phosphor_centers = (555.0, 575.0, 595.0)
    blue = gaussian_spectrum(wavelengths, blue_centers[variant], 13.0)
    phosphor = gaussian_spectrum(wavelengths, phosphor_centers[variant], 75.0)
    density = (0.65 - 0.12 * variant) * blue + (0.35 + 0.12 * variant) * phosphor
    return density / np.trapz(density, wavelengths)


def build_dictionary(wavelengths: np.ndarray) -> np.ndarray:
    atoms = [daylight_spectrum(wavelengths, x) for x in (5.0, 15.0, 30.0, 45.0, 60.0, 80.0)]
    atoms.extend(planck_spectrum(wavelengths, x) for x in (2700.0, 4000.0, 6500.0))
    atoms.extend(fluorescent_spectrum(wavelengths, x) for x in range(4))
    atoms.extend(led_spectrum(wavelengths, x) for x in range(3))
    return np.stack(atoms, axis=1)


def integration_matrix(grid: SpectralGrid, dictionary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grid.validate()
    delta = np.gradient(grid.wavelengths)
    sensor = (grid.responsivity * delta[None, :]) @ dictionary
    alpha = (grid.action_spectra * delta[None, :]) @ dictionary
    return sensor, alpha


def smoothness_matrix(size: int) -> np.ndarray:
    matrix = np.zeros((size - 2, size), dtype=np.float64)
    for row in range(size - 2):
        matrix[row, row] = 1.0
        matrix[row, row + 1] = -2.0
        matrix[row, row + 2] = 1.0
    return matrix


class SpectralDictionary:
    def __init__(self, sensor_matrix: np.ndarray, alpha_matrix: np.ndarray, regularization: float = 1e-2) -> None:
        if sensor_matrix.shape[0] != 3:
            raise ValueError("sensor matrix must have three rows")
        if alpha_matrix.shape[0] != 5:
            raise ValueError("alpha matrix must have five rows")
        if sensor_matrix.shape[1] != alpha_matrix.shape[1]:
            raise ValueError("dictionary dimensions differ")
        self.sensor_matrix = sensor_matrix.astype(np.float64)
        self.alpha_matrix = alpha_matrix.astype(np.float64)
        self.regularization = float(regularization)
        self.penalty = smoothness_matrix(sensor_matrix.shape[1])

    @property
    def atoms(self) -> int:
        return self.sensor_matrix.shape[1]

    def coefficients(self, colour: np.ndarray) -> np.ndarray:
        colour = np.asarray(colour, dtype=np.float64)
        if colour.shape != (3,):
            raise ValueError("colour must have shape [3]")
        augmented_matrix = np.concatenate((self.sensor_matrix, self.regularization * self.penalty), axis=0)
        augmented_target = np.concatenate((np.maximum(colour, 0.0), np.zeros(self.penalty.shape[0])))
        coefficients, _ = nnls(augmented_matrix, augmented_target)
        return coefficients

    def alpha_opic(self, colour: np.ndarray) -> np.ndarray:
        return self.alpha_matrix @ self.coefficients(colour)

    def metamer_bounds(self, colour: np.ndarray, action: int, tolerance: float = 1e-5) -> tuple[float, float]:
        from scipy.optimize import linprog

        target = np.asarray(colour, dtype=np.float64)
        constraints = np.concatenate((self.sensor_matrix, -self.sensor_matrix), axis=0)
        upper = np.concatenate((target + tolerance, -target + tolerance))
        objective = self.alpha_matrix[action]
        lower_result = linprog(objective, A_ub=constraints, b_ub=upper, bounds=(0.0, None), method="highs")
        upper_result = linprog(-objective, A_ub=constraints, b_ub=upper, bounds=(0.0, None), method="highs")
        if not lower_result.success or not upper_result.success:
            raise RuntimeError("metamer optimization failed")
        return float(lower_result.fun), float(-upper_result.fun)


class ChromaticityLookup(nn.Module):
    def __init__(self, table: Tensor, alpha_matrix: Tensor) -> None:
        super().__init__()
        if table.ndim != 3:
            raise ValueError("table must have shape [bins, bins, atoms]")
        self.register_buffer("table", table.to(torch.int8))
        self.register_buffer("alpha_matrix", alpha_matrix.to(torch.float32))
        self.register_buffer("coefficient_scale", torch.tensor(127.0))

    @classmethod
    def from_dictionary(cls, dictionary: SpectralDictionary, bins: int = 64) -> "ChromaticityLookup":
        table = np.empty((bins, bins, dictionary.atoms), dtype=np.int8)
        for first in range(bins):
            for second in range(bins):
                ratios = np.array(((first + 0.5) / bins * 4.0, 1.0, (second + 0.5) / bins * 4.0))
                coefficients = dictionary.coefficients(ratios)
                peak = max(float(coefficients.max()), 1e-12)
                table[first, second] = np.round(coefficients / peak * 127.0).astype(np.int8)
        return cls(torch.from_numpy(table), torch.from_numpy(dictionary.alpha_matrix))

    def forward(self, colour: Tensor) -> Tensor:
        if colour.shape[-1] != 3:
            raise ValueError("last colour dimension must equal three")
        magnitude = colour[..., 1].clamp_min(1e-8)
        bins = self.table.shape[0]
        first = torch.clamp((colour[..., 0] / magnitude * bins / 4.0).long(), 0, bins - 1)
        second = torch.clamp((colour[..., 2] / magnitude * bins / 4.0).long(), 0, bins - 1)
        coefficients = self.table[first, second].float() / self.coefficient_scale
        coefficients = coefficients * magnitude.unsqueeze(-1)
        return coefficients @ self.alpha_matrix.transpose(0, 1)


def log_compress(alpha: Tensor, floor: float = 1e-4) -> Tensor:
    return torch.log1p(alpha.clamp_min(floor))


def timeline_features(alpha: Tensor, dark_threshold: float = 1.0) -> Tensor:
    if alpha.ndim != 3 or alpha.shape[-1] != 5:
        raise ValueError("alpha must have shape [batch, time, five]")
    melanopic = alpha[..., 4]
    daylight = melanopic.clamp_min(1e-6).log().mean(dim=1).exp()
    dark = (melanopic < dark_threshold).float()
    dark_fraction = dark.mean(dim=1)
    dark_variance = dark.var(dim=1, unbiased=False)
    first = dark.argmax(dim=1).float() / melanopic.shape[1]
    last = 1.0 - dark.flip(1).argmax(dim=1).float() / melanopic.shape[1]
    return torch.stack((daylight, dark_fraction, first, last, dark_variance), dim=-1)


def exposure_statistics(lux: Tensor, bin_minutes: float = 10.0) -> Tensor:
    above_1000 = (lux >= 1000.0).float().sum(dim=-1) * bin_minutes
    above_3000 = (lux >= 3000.0).float().sum(dim=-1) * bin_minutes
    active = lux >= 2000.0
    previous = torch.nn.functional.pad(active[..., :-1], (1, 0), value=False)
    starts = (active & ~previous).float().sum(dim=-1)
    cumulative = lux.sum(dim=-1) * bin_minutes
    return torch.stack((above_1000, above_3000, starts, cumulative), dim=-1)
