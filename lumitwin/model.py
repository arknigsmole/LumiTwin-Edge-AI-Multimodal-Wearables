from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .controller import fake_quantize
from .light import exposure_statistics, timeline_features


class CausalDepthwiseBlock(nn.Module):
    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        self.padding = 2 * dilation
        self.depthwise = nn.Conv1d(width, width, 3, dilation=dilation, padding=self.padding, groups=width)
        self.pointwise = nn.Conv1d(width, width * 2, 1)
        self.normalization = nn.GroupNorm(1, width)

    def forward(self, inputs: Tensor) -> Tensor:
        filtered = self.depthwise(inputs)[..., : inputs.shape[-1]]
        value, gate = self.pointwise(filtered).chunk(2, dim=1)
        return self.normalization(inputs + value * torch.sigmoid(gate))


class LightTimelineEncoder(nn.Module):
    def __init__(self, channels: int = 5, width: int = 48) -> None:
        super().__init__()
        self.projection = nn.Conv1d(channels, width, 1)
        self.blocks = nn.ModuleList(CausalDepthwiseBlock(width, 2**index) for index in range(10))
        self.feature_projection = nn.Linear(5, width)

    def forward(self, timeline: Tensor) -> tuple[Tensor, Tensor]:
        encoded = self.projection(timeline.transpose(1, 2))
        for block in self.blocks:
            encoded = block(encoded)
        summary = encoded[..., -1]
        features = timeline_features(timeline)
        summary = summary + self.feature_projection(features)
        return encoded.transpose(1, 2), summary


class InvertedResidual(nn.Module):
    def __init__(self, inputs: int, outputs: int, expansion: int = 2, stride: int = 1) -> None:
        super().__init__()
        hidden = inputs * expansion
        self.body = nn.Sequential(nn.Conv1d(inputs, hidden, 1, bias=False), nn.BatchNorm1d(hidden), nn.SiLU(), nn.Conv1d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False), nn.BatchNorm1d(hidden), nn.SiLU(), nn.Conv1d(hidden, outputs, 1, bias=False), nn.BatchNorm1d(outputs))
        self.skip = nn.Identity() if inputs == outputs and stride == 1 else nn.Conv1d(inputs, outputs, 1, stride=stride)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.body(inputs) + self.skip(inputs)


class SensorEncoder(nn.Module):
    def __init__(self, channels: int = 6, width: int = 48) -> None:
        super().__init__()
        self.blocks = nn.Sequential(InvertedResidual(channels, 24, 2, 2), InvertedResidual(24, 32, 2, 2), InvertedResidual(32, width, 2, 2))
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.output = nn.Linear(width, width)

    def forward(self, epoch: Tensor) -> tuple[Tensor, Tensor]:
        local = self.blocks(epoch.transpose(1, 2)).transpose(1, 2)
        global_features, _ = self.attention(local, local, local, need_weights=False)
        fused = local + global_features
        return fused, self.output(fused.mean(dim=1))


class GatedFusion(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.gate = nn.Linear(width * 2, width)
        self.value = nn.Linear(width * 2, width)

    def forward(self, light: Tensor, sensor: Tensor) -> Tensor:
        combined = torch.cat((light, sensor), dim=-1)
        return torch.sigmoid(self.gate(combined)) * self.value(combined)


class TaskAdapter(nn.Module):
    def __init__(self, width: int, rank: int = 8) -> None:
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        self.normalization = nn.LayerNorm(width)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.normalization(inputs + self.up(self.down(inputs)))


class CircadianHead(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.observation = nn.Linear(width, 3)
        self.sleep = nn.Linear(width, 5)
        self.likelihood = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 2))
        self.drive_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        likelihood = self.likelihood(features)
        return {"particle_observation": self.observation(features), "sleep_logits": self.sleep(features), "phase_location": likelihood[..., 0], "phase_scale": torch.nn.functional.softplus(likelihood[..., 1]), "drive_scale": self.drive_scale.exp()}


class ExposureHead(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.regressor = nn.Sequential(nn.Linear(width + 4, width), nn.SiLU(), nn.Linear(width, 4))

    def forward(self, features: Tensor, lux: Tensor) -> Tensor:
        exact = exposure_statistics(lux)
        return self.regressor(torch.cat((features, exact), dim=-1))


class RefractiveHead(nn.Module):
    def __init__(self, width: int, covariates: int = 8) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.pool = nn.MultiheadAttention(width, 4, batch_first=True)
        self.embedding = nn.Sequential(nn.Linear(width + covariates, width), nn.SiLU())
        self.onset = nn.Linear(width, 1)
        self.spherical_equivalent = nn.Linear(width, 1)

    def forward(self, sequence: Tensor, covariates: Tensor) -> tuple[Tensor, Tensor]:
        query = self.query.expand(sequence.shape[0], -1, -1)
        pooled, _ = self.pool(query, sequence, sequence, need_weights=False)
        embedded = self.embedding(torch.cat((pooled[:, 0], covariates), dim=-1))
        return self.onset(embedded).squeeze(-1), self.spherical_equivalent(embedded).squeeze(-1)


@dataclass
class LumiTwinOutput:
    sleep_logits: Tensor
    particle_observation: Tensor
    phase_location: Tensor
    phase_scale: Tensor
    drive_scale: Tensor
    exposure: Tensor
    onset_logit: Tensor
    spherical_equivalent: Tensor
    representation: Tensor


class LumiTwin(nn.Module):
    def __init__(self, sensor_channels: int = 6, covariates: int = 8, width: int = 48, rank: int = 8) -> None:
        super().__init__()
        self.light_encoder = LightTimelineEncoder(5, width)
        self.sensor_encoder = SensorEncoder(sensor_channels, width)
        self.fusion = GatedFusion(width)
        self.adapters = nn.ModuleDict({name: TaskAdapter(width, rank) for name in ("circadian", "exposure", "refractive")})
        self.circadian = CircadianHead(width)
        self.exposure = ExposureHead(width)
        self.refractive = RefractiveHead(width, covariates)

    def forward(self, timeline: Tensor, sensor_epoch: Tensor, lux: Tensor, covariates: Tensor, bits: int = 8) -> LumiTwinOutput:
        light_sequence, light_summary = self.light_encoder(timeline)
        sensor_sequence, sensor_summary = self.sensor_encoder(sensor_epoch)
        representation = fake_quantize(self.fusion(light_summary, sensor_summary), bits)
        circadian = self.circadian(self.adapters["circadian"](representation))
        exposure = self.exposure(self.adapters["exposure"](representation), lux)
        refractive_sequence = self.adapters["refractive"](light_sequence)
        onset, spherical = self.refractive(refractive_sequence, covariates)
        return LumiTwinOutput(circadian["sleep_logits"], circadian["particle_observation"], circadian["phase_location"], circadian["phase_scale"], circadian["drive_scale"], exposure, onset, spherical, representation)


def parameter_sharing(model: LumiTwin) -> float:
    total = sum(parameter.numel() for parameter in model.parameters())
    shared = sum(parameter.numel() for name, parameter in model.named_parameters() if not any(token in name for token in ("adapters", "circadian", "exposure", "refractive")))
    return shared / total
