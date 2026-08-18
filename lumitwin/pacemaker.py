from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class PacemakerParameters:
    tau_x: float = 24.2
    exponent: float = 0.5
    beta: float = 0.0075
    gain: float = 33.75
    stiffness: float = 0.55
    cubic: float = 1.0 / 3.0
    damping: float = 0.13


def photic_drive(medi: Tensor, scale: Tensor, reference: float, exponent: float) -> Tensor:
    return scale * (medi.clamp_min(0.0) / reference).pow(exponent)


def augmented_generator(state: Tensor, drive: Tensor, parameters: PacemakerParameters) -> Tensor:
    x = state[..., 0]
    y = state[..., 1]
    radial = x.square() + y.square()
    restoring = parameters.stiffness * (1.0 - parameters.cubic * radial)
    batch_shape = state.shape[:-1]
    matrix = torch.zeros(*batch_shape, 3, 3, dtype=state.dtype, device=state.device)
    frequency = 2.0 * torch.pi / parameters.tau_x
    matrix[..., 0, 0] = parameters.damping * restoring
    matrix[..., 0, 1] = frequency
    matrix[..., 0, 2] = parameters.gain * drive
    matrix[..., 1, 0] = -frequency
    matrix[..., 1, 1] = parameters.damping * restoring
    return matrix


def magnus_second_order(state: Tensor, drive_start: Tensor, drive_end: Tensor, dt_hours: float, parameters: PacemakerParameters) -> Tensor:
    first = augmented_generator(state, drive_start, parameters)
    predictor = torch.cat((state, torch.ones_like(state[..., :1])), dim=-1)
    provisional = torch.matrix_exp(first * dt_hours) @ predictor.unsqueeze(-1)
    second_state = provisional.squeeze(-1)[..., :2]
    second = augmented_generator(second_state, drive_end, parameters)
    omega_one = 0.5 * dt_hours * (first + second)
    commutator = second @ first - first @ second
    omega_two = dt_hours * dt_hours / 12.0 * commutator
    propagated = torch.matrix_exp(omega_one + 0.5 * omega_two) @ predictor.unsqueeze(-1)
    return propagated.squeeze(-1)[..., :2]


def photoreceptor_update(activation: Tensor, drive: Tensor, beta: float, dt_minutes: float) -> Tensor:
    steady = drive / (drive + beta).clamp_min(1e-8)
    decay = torch.exp(-(drive + beta) * dt_minutes)
    return steady + (activation - steady) * decay


class AliasTable:
    def __init__(self, probabilities: Tensor) -> None:
        probabilities = probabilities / probabilities.sum().clamp_min(1e-12)
        size = probabilities.numel()
        scaled = probabilities * size
        small = torch.nonzero(scaled < 1.0, as_tuple=False).flatten().tolist()
        large = torch.nonzero(scaled >= 1.0, as_tuple=False).flatten().tolist()
        probability = torch.ones(size, dtype=torch.float32, device=probabilities.device)
        alias = torch.arange(size, dtype=torch.long, device=probabilities.device)
        while small and large:
            low = small.pop()
            high = large.pop()
            probability[low] = scaled[low]
            alias[low] = high
            scaled[high] = scaled[high] - (1.0 - scaled[low])
            if scaled[high] < 1.0:
                small.append(high)
            else:
                large.append(high)
        self.probability = torch.round(probability * 255.0).to(torch.uint8)
        self.alias = alias

    def draw(self, count: int, generator: torch.Generator | None = None) -> Tensor:
        size = self.alias.numel()
        column = torch.randint(size, (count,), device=self.alias.device, generator=generator)
        threshold = torch.randint(256, (count,), device=self.alias.device, generator=generator)
        accept = threshold <= self.probability[column]
        return torch.where(accept, column, self.alias[column])


class ParticleFilter(nn.Module):
    def __init__(self, particles: int = 256, weight_floor: float = 1e-3, dt_seconds: float = 30.0) -> None:
        super().__init__()
        angles = torch.linspace(-torch.pi, torch.pi, particles + 1)[:-1]
        initial = torch.stack((angles.cos(), angles.sin()), dim=-1)
        self.register_buffer("initial_particles", initial)
        self.particles_count = particles
        self.weight_floor = weight_floor
        self.dt_hours = dt_seconds / 3600.0
        self.parameters = PacemakerParameters()

    def initialize(self, batch: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        particles = self.initial_particles.to(device).unsqueeze(0).expand(batch, -1, -1).clone()
        weights = torch.full((batch, self.particles_count), 1.0 / self.particles_count, device=device)
        activation = torch.zeros(batch, self.particles_count, device=device)
        return particles, weights, activation

    def step(self, particles: Tensor, weights: Tensor, activation: Tensor, medi: Tensor, observation: Tensor, scale: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        drive = photic_drive(medi, scale, 100.0, self.parameters.exponent)
        expanded_drive = drive.unsqueeze(-1).expand_as(weights)
        next_activation = photoreceptor_update(activation, expanded_drive, self.parameters.beta, self.dt_hours * 60.0)
        next_particles = magnus_second_order(particles, activation, next_activation, self.dt_hours, self.parameters)
        predicted = torch.stack((next_particles[..., 0], next_particles[..., 1], next_activation), dim=-1)
        residual = observation.unsqueeze(1) - predicted
        likelihood = torch.exp(-0.5 * residual.square().sum(dim=-1))
        normalized = weights * likelihood
        normalized = normalized / normalized.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        normalized = normalized.clamp_min(self.weight_floor)
        normalized = normalized / normalized.sum(dim=-1, keepdim=True)
        resampled_particles = []
        resampled_activation = []
        for row in range(normalized.shape[0]):
            indices = AliasTable(normalized[row]).draw(self.particles_count)
            resampled_particles.append(next_particles[row, indices])
            resampled_activation.append(next_activation[row, indices])
        output_particles = torch.stack(resampled_particles)
        output_activation = torch.stack(resampled_activation)
        output_weights = torch.full_like(normalized, 1.0 / self.particles_count)
        phase = torch.atan2((normalized * next_particles[..., 1]).sum(dim=-1), (normalized * next_particles[..., 0]).sum(dim=-1))
        return output_particles, output_weights, output_activation, phase


def circular_error(prediction: Tensor, target: Tensor) -> Tensor:
    difference = torch.atan2(torch.sin(prediction - target), torch.cos(prediction - target))
    return difference.abs()


def circular_mean(angles: Tensor, weights: Tensor | None = None, dim: int = -1) -> Tensor:
    if weights is None:
        weights = torch.ones_like(angles)
    sine = (weights * angles.sin()).sum(dim=dim)
    cosine = (weights * angles.cos()).sum(dim=dim)
    return torch.atan2(sine, cosine)
