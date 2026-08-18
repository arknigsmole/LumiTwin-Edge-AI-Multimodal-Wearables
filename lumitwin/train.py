import argparse
import logging
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .checkpoint import load, save_atomic, set_seed
from .data import MulticohortDataset, collate_disjoint
from .model import LumiTwin
from .objectives import DisjointLabelObjective


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="lumitwin-train")
    parser.add_argument("--config", type=Path, default=Path("config/main.yaml"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def epoch_pass(model: LumiTwin, loader: DataLoader[dict[str, torch.Tensor]], objective: DisjointLabelObjective, device: torch.device, optimizer: torch.optim.Optimizer | None) -> float:
    model.train(optimizer is not None)
    total = 0.0
    count = 0
    for cpu_batch in loader:
        batch = move(cpu_batch, device)
        with torch.set_grad_enabled(optimizer is not None):
            output = model(batch["timeline"], batch["sensor"], batch["lux"], batch["covariates"])
            terms = objective(output, batch)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                terms.total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total += float(terms.total.detach()) * batch["timeline"].shape[0]
        count += batch["timeline"].shape[0]
    return total / max(count, 1)


def main() -> None:
    args = arguments()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    seed = int(config["seed"])
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training = MulticohortDataset(args.manifest, args.data_root, "train", seed, int(config["timeline_bins"]))
    validation = MulticohortDataset(args.manifest, args.data_root, "validation", seed, int(config["timeline_bins"]))
    training_loader = DataLoader(training, batch_size=int(config["batch_size"]), shuffle=True, num_workers=4, pin_memory=device.type == "cuda", collate_fn=collate_disjoint)
    validation_loader = DataLoader(validation, batch_size=int(config["batch_size"]), shuffle=False, num_workers=4, pin_memory=device.type == "cuda", collate_fn=collate_disjoint)
    model = LumiTwin(width=int(config["trunk_width"]), rank=int(config["adapter_rank"])).to(device)
    objective = DisjointLabelObjective().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    start = 0
    best = float("inf")
    if args.resume is not None:
        start, restored_seed, best = load(args.resume, model, optimizer, device)
        if restored_seed != seed:
            raise ValueError("checkpoint seed differs from configuration")
        start += 1
    stale = 0
    for epoch in range(start, int(config["epochs"])):
        training_loss = epoch_pass(model, training_loader, objective, device, optimizer)
        validation_loss = epoch_pass(model, validation_loader, objective, device, None)
        logging.info("epoch=%d training_loss=%.6f validation_loss=%.6f", epoch, training_loss, validation_loss)
        save_atomic(args.output / "latest.pt", model, optimizer, epoch, seed, best)
        if validation_loss < best:
            best = validation_loss
            stale = 0
            save_atomic(args.output / "best.pt", model, optimizer, epoch, seed, best)
        else:
            stale += 1
        if stale >= int(config["early_stopping_patience"]):
            break


if __name__ == "__main__":
    main()
