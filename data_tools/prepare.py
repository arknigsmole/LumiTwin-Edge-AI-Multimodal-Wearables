import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="lumitwin-prepare")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--site", required=True)
    return parser.parse_args()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def numeric(frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    missing = set(names).difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return frame[names].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)


def process(source: Path, output: Path, site: str) -> dict[str, object]:
    frame = pd.read_csv(source)
    colour = numeric(frame, ["red", "green", "blue"])
    sensor = numeric(frame, ["accel_x", "accel_y", "accel_z", "ppg", "temperature", "activity"])
    lux = numeric(frame, ["lux"])[:, 0]
    if len(frame) < 400:
        raise ValueError("each file must contain at least 400 samples")
    subject = str(frame["subject"].iloc[0])
    folder = output / subject
    folder.mkdir(parents=True, exist_ok=True)
    colour_path = folder / "colour.npy"
    sensor_path = folder / "sensor.npy"
    lux_path = folder / "lux.npy"
    np.save(colour_path, colour, allow_pickle=False)
    np.save(sensor_path, sensor, allow_pickle=False)
    np.save(lux_path, lux, allow_pickle=False)
    return {"subject": subject, "site": site, "colour_path": str(colour_path.relative_to(output)), "sensor_path": str(sensor_path.relative_to(output)), "lux_path": str(lux_path.relative_to(output)), "source_sha256": digest(source)}


def main() -> None:
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = sorted(args.input.glob("*.csv"))
    if not sources:
        raise ValueError("input contains no CSV records")
    manifest = pd.DataFrame(process(source, args.output, args.site) for source in sources)
    manifest.to_csv(args.output / "manifest.csv", index=False)


if __name__ == "__main__":
    main()
