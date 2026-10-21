"""TorchXRayVision DenseNet121 + U-Ignore: 50-epoch training.

- No early stopping.
- Select best_model.pth by minimum validation loss.
- Save last_model.pth after every epoch and periodic checkpoints every 10 epochs.
- Optionally resume from a checkpoint created by this script.
"""
from __future__ import annotations

import csv
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
import torchxrayvision as xrv
from torch import nn
from torch.utils.data import DataLoader, Subset

# This file belongs in <project>/torchxrayvision/train_xrv.py
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from cxr_preprocessing_uignore import (
    TARGET_LABELS,
    build_dataloaders,
    check_uignore_batch,
)
from train_densenet121_uignore_full import run_epoch
from cxr_xrv_adapter import prepare_xrv_images

CSV_DIR = PROJECT_DIR / "data" / "processed"
IMAGE_ROOT = Path("/home/team30103/physionet_data")
OUTPUT_ROOT = PROJECT_DIR / "torchxrayvision" / "runs"

PILOT = False
EPOCHS = 2 if PILOT else 50
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SAVE_EVERY = 10
SEED = 42
NUM_WORKERS = 0

# For a new 50-epoch run, keep None.
# To resume later, point to a checkpoint saved by THIS script, for example:
# RESUME_FROM = OUTPUT_ROOT / "full_50ep_20261021_120000_000000" / "last_model.pth"
RESUME_FROM = None


class XRVUIgnore(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = xrv.models.DenseNet(weights="densenet121-res224-chex")
        in_features = self.backbone.classifier.in_features
        self.classifier = nn.Linear(in_features, 3 * 2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = prepare_xrv_images(images)
        features = self.backbone.features2(images)
        return self.classifier(features).reshape(-1, 3, 2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_checkpoint(path: Path, payload: dict) -> None:
    """Write atomically to avoid a partially overwritten checkpoint."""
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def make_checkpoint(model, optimizer, epoch, best_loss, best_epoch,
                    val_metrics, train_loader, config) -> dict:
    generator = getattr(train_loader, "generator", None)
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_loss,
        "best_epoch": best_epoch,
        "val_metrics": val_metrics,
        "config": config,
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all(),
        "train_loader_generator_state": generator.get_state() if generator else None,
    }


def main() -> None:
    set_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU가 인식되지 않습니다.")
    device = torch.device("cuda")
    print("Device:", device, flush=True)
    print("Mode:", "PILOT" if PILOT else "FULL", flush=True)
    print("Epochs:", EPOCHS, "| Early stopping: OFF", flush=True)

    loaders = build_dataloaders(
        csv_dir=CSV_DIR,
        image_root=IMAGE_ROOT,
        image_layout="flat",
        image_size=224,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        seed=SEED,
    )
    train_loader = loaders["train"]
    val_loader = loaders["validate"]

    if PILOT:
        rng = random.Random(SEED)
        train_ids = rng.sample(range(len(train_loader.dataset)), 128)
        val_ids = rng.sample(range(len(val_loader.dataset)), 64)
        train_loader = DataLoader(
            Subset(train_loader.dataset, train_ids),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
            generator=torch.Generator().manual_seed(SEED),
        )
        val_loader = DataLoader(
            Subset(val_loader.dataset, val_ids),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        )

    print("Train images:", len(train_loader.dataset), flush=True)
    print("Val images:", len(val_loader.dataset), flush=True)
    images, labels = next(iter(train_loader))
    check_uignore_batch(images, labels)

    model = XRVUIgnore().to(device)
    model.eval()  # Prevent BatchNorm statistics from changing during a shape test.
    with torch.no_grad():
        output = model(images.to(device))
    print("Model output shape:", output.shape, flush=True)
    assert output.shape == (len(images), 3, 2)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    config = {
        "architecture": "TorchXRayVision DenseNet121",
        "weights": "densenet121-res224-chex",
        "label_strategy": "U-Ignore",
        "target_labels": list(TARGET_LABELS),
        "selection_metric": "minimum validation loss",
        "pilot": PILOT,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "save_every": SAVE_EVERY,
        "seed": SEED,
        "preprocessing": "cxr_preprocessing.py + cxr_preprocessing_uignore.py + cxr_xrv_adapter.py",
        "early_stopping": False,
    }

    best_loss, best_epoch, start_epoch = math.inf, 0, 1
    history = []
    if RESUME_FROM is None:
        prefix = "pilot" if PILOT else "full_50ep"
        run_dir = OUTPUT_ROOT / (prefix + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        ckpt_path = Path(RESUME_FROM).expanduser().resolve()
        if not ckpt_path.is_file():
            raise FileNotFoundError(ckpt_path)
        run_dir = ckpt_path.parent
        # Load only checkpoints you have created and trust.
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        saved_config = checkpoint["config"]
        for key in ("weights", "pilot", "batch_size", "learning_rate", "weight_decay", "seed"):
            if saved_config[key] != config[key]:
                raise ValueError(f"이어 학습 설정 불일치: {key}")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_loss = float(checkpoint["best_val_loss"])
        best_epoch = int(checkpoint["best_epoch"])
        start_epoch = int(checkpoint["epoch"]) + 1
        history_path = run_dir / "history.csv"
        if history_path.exists():
            with history_path.open(newline="", encoding="utf-8") as f:
                history = [row for row in csv.DictReader(f)
                           if int(row["epoch"]) < start_epoch]
        # Restore randomness after model construction and the preliminary batch check.
        random.setstate(checkpoint["python_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        generator = getattr(train_loader, "generator", None)
        if generator is not None and checkpoint["train_loader_generator_state"] is not None:
            generator.set_state(checkpoint["train_loader_generator_state"])
        print(f"Resume: Epoch {start_epoch}부터 ({ckpt_path})", flush=True)

    for epoch in range(start_epoch, EPOCHS + 1):
        train = run_epoch(model, train_loader, device, optimizer)
        val = run_epoch(model, val_loader, device)

        history.append({
            "epoch": epoch,
            "train_loss": train["loss"],
            "val_loss": val["loss"],
            "train_macro_f1": train["macro_f1"],
            "val_macro_f1": val["macro_f1"],
            "train_valid_labels": train["valid_label_count"],
            "val_valid_labels": val["valid_label_count"],
        })
        with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)

        print(
            f"Epoch {epoch}/{EPOCHS} | train loss {train['loss']:.4f} | "
            f"val loss {val['loss']:.4f} | val macro F1 {val['macro_f1']:.4f}",
            flush=True,
        )

        improved = val["loss"] < best_loss
        if improved:
            best_loss = val["loss"]
            best_epoch = epoch

        payload = make_checkpoint(
            model, optimizer, epoch, best_loss, best_epoch, val, train_loader, config
        )
        # Recover from the last completed epoch even if the job disconnects.
        save_checkpoint(run_dir / "last_model.pth", payload)

        if improved:
            save_checkpoint(run_dir / "best_model.pth", payload)
            (run_dir / "best_val_metrics.json").write_text(
                json.dumps(val, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print("Best model saved!", flush=True)
        else:
            print("Validation Loss 개선 없음 (학습 계속)", flush=True)

        if epoch % SAVE_EVERY == 0:
            save_checkpoint(run_dir / f"checkpoint_epoch_{epoch}.pth", payload)
            print(f"Checkpoint saved: Epoch {epoch}", flush=True)

    print("Training finished!", flush=True)
    print(f"Best epoch: {best_epoch} | Best val loss: {best_loss:.4f}", flush=True)
    print("Results:", run_dir, flush=True)


if __name__ == "__main__":
    main()
