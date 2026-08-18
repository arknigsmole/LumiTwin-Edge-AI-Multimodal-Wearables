# Edge AI–Enabled Multimodal Wearables for Circadian and Photic-Exposure Phenotyping from Ambient Light

LumiTwin maps wrist colour photometry and physiological signals to a shared representation for sleep state, circadian phase, photic-exposure indices, and refractive outcomes. The implementation contains the three-channel physics-informed light model, a 1,008-bin causal light timeline, a separable physiological encoder, rank-8 task adapters, the second-order Magnus pacemaker update, a 256-particle alias-resampled filter, and adaptive precision and sensor duty control.

## Scope

The released training path covers the measurements and objectives that can be constructed from the listed corpora. The paper does not identify a valid public repository for its paediatric refractive cohort, and its cited item is a statistics textbook. Refractive labels therefore require a user-supplied, lawfully obtained manifest. Near-corneal spectral data described as a 2026 corpus also lacks a verifiable repository record and is not linked.

## Installation

Python 3.11 is required.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps .
```

Conda users can run `conda env create -f environment.yml`. A container can be built with `docker build -t lumitwin .`.

## Data

Verified dataset landing pages and access conditions are collected in `dataset_links.txt`. MESA Sleep and HCHS/SOL Sueño require approval and a data-use agreement. SandD is controlled and restricted to non-commercial use. PPG-DaLiA is distributed under CC BY 4.0. These sources do not form a fully labeled cohort; labels remain disjoint across sources.

Raw participant exports are never committed. A source CSV must contain `subject`, `red`, `green`, `blue`, `accel_x`, `accel_y`, `accel_z`, `ppg`, `temperature`, `activity`, and `lux`. Convert a directory with:

```bash
lumitwin-prepare --input datasets/site_a/raw --output datasets/site_a/processed --site site_a
```

The command creates NumPy arrays and a manifest with a SHA-256 digest for each source record. Merge site manifests while retaining the `site` field, then add available labels as `sleep`, `phase`, `exposure_0` through `exposure_3`, `onset`, and `spherical_equivalent`. Missing labels are represented as empty values.

## Training

The manuscript reports AdamW, learning rate `3e-4`, batch size `64`, `200` epochs, and early-stopping patience `20`; these are the defaults in `config/main.yaml`.

```bash
lumitwin-train --manifest datasets/manifest.csv --data-root datasets --output outputs/main
```

Resume with `--resume outputs/main/latest.pt`. Checkpoint writes are atomic and retain optimizer and random-generator states.

## Architecture

PIALM fits non-negative coefficients over sixteen illuminant atoms and maps them into five CIE S 026 α-opic channels. The chromaticity representation uses a 64 by 64 table. Light samples are log-compressed and aggregated into ten-minute bins over seven days. The temporal encoder has ten depthwise-separable causal blocks with dilations from 1 through 512 and width 48. A compact convolutional sensor encoder processes the 30-second epoch. Gated fusion feeds task-specific rank-8 adapters.

The circadian branch evolves the two-dimensional limit-cycle state with a second-order Magnus propagator and filters it with 256 particles. Weights are clamped at `1e-3`, normalized, converted to an eight-bit Vose alias table, and rebuilt at every epoch. The exposure branch estimates time above 1,000 and 3,000 lux, bouts above 2,000 lux, and cumulative exposure. The refractive branch pools the seven-day representation with subject covariates.

AQDC enumerates bit widths 4 and 8 and sampling duties 1, 1/2, and 1/4. Its virtual queue chooses the operating point from posterior uncertainty, light variability, energy, and distortion.

## Reported compute and limitations

The manuscript does not report training GPU model, count, VRAM, storage, numeric precision, warmup, weight decay, scheduler, or training wall-clock. No substitute values are presented as reported facts. Deployment measurements target generic Cortex-M7, Cortex-M33, and Cortex-A55 classes. The reported Cortex-M7 AQDC operating point is 5.6 ms latency, 178 kB peak SRAM, and 0.89 mJ per 30-second epoch.

The input colour bands in MESA and HCHS/SOL are not traceably calibrated spectroradiometers. Their α-opic outputs are proxies. Absolute α-opic evaluation requires measured spectral irradiance. Three colour channels cannot identify arbitrary spectra and are expected to fail for narrowband sources.

## Expected measurements

The reported full configuration has sleep-wake κ `0.641`, phase disagreement `21.6 min`, exposure error `9.6 min/day`, bout error `0.41/day`, 120-minute exposure AUC `0.931`, refractive-onset AUROC `0.902`, and spherical-equivalent MAE `0.388 D`. These numbers are reference targets, not guarantees for independently prepared cohorts.
