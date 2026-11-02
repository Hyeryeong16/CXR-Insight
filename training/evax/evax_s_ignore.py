"""EVA-X-S + U-Ignore full dataset training.

Train: 9,999 images
Validation: 1,077 images
Test: 2,284 images (not used for training)
"""
from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
import sys

EVA_X_DIR = Path.home() / "eva_x_project"
sys.path.insert(0, str(EVA_X_DIR))
from eva_x import eva_x_small_patch16

import cxr_preprocessing as preprocessing_base
preprocessing_base.MEAN = (0.49185243,) * 3
preprocessing_base.STD = (0.28509309,) * 3


from cxr_preprocessing_uignore import (
    IGNORE_INDEX, TARGET_LABELS, build_dataloaders, check_uignore_batch,
)

PROJECT_DIR = Path(__file__).resolve().parent
CSV_DIR = PROJECT_DIR / "data" / "processed"
IMAGE_ROOT = Path("/home/team30103/physionet_data")
OUTPUT_ROOT = PROJECT_DIR / "runs" / "evax_s_uignore_full"
CLASS_NAMES = ("negative", "positive")


@dataclass
class TrainConfig:
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 3
    batch_size: int = 4
    seed: int = 42
    image_size: int = 224
    num_workers: int = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class EVAXSUIgnore(nn.Module):
    def __init__(self):
        super().__init__()
        weights = (
            Path.home() / "eva_x_project" / "weights"
            / "eva_x_small_patch16_merged520k_mim.pt"
        )
        if not weights.is_file():
            raise FileNotFoundError(weights)

        self.backbone = eva_x_small_patch16(pretrained=str(weights))
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Linear(in_features, 3 * 2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images).reshape(-1, 3, 2)



def masked_loss_sum(logits: torch.Tensor, labels: torch.Tensor):
    """Return differentiable SUM of valid CE terms and number of valid labels.

    A completely ignored batch has count=0; caller must skip backward/step.
    """
    losses = F.cross_entropy(
        logits.reshape(-1, 2), labels.reshape(-1),
        ignore_index=IGNORE_INDEX, reduction="none",
    )
    valid = labels.reshape(-1) != IGNORE_INDEX
    return losses[valid].sum(), int(valid.sum().item())


def run_epoch(model, loader, device, optimizer=None) -> dict:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss_sum, total_valid, total_images, skipped = 0.0, 0, 0, 0
    # confusion[3 findings, true 2 classes, predicted 2 classes]
    confusion = torch.zeros((3, 2, 2), dtype=torch.long)

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            check_uignore_batch(images, labels)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss_sum, valid_count = masked_loss_sum(logits, labels)
            if valid_count == 0:
                skipped += 1
                total_images += len(images)
                continue
            loss = loss_sum / valid_count
            if not torch.isfinite(loss).item():
                raise RuntimeError("Loss가 NaN/Inf입니다.")
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss_sum += float(loss_sum.detach().item())
            total_valid += valid_count
            total_images += len(images)

            predictions = logits.detach().argmax(dim=-1).cpu()
            targets = labels.detach().cpu()
            for i in range(3):
                valid = targets[:, i] != IGNORE_INDEX
                true_i = targets[valid, i]
                pred_i = predictions[valid, i]
                index = true_i * 2 + pred_i
                confusion[i] += torch.bincount(index, minlength=4).reshape(2, 2)

    if total_valid == 0:
        raise RuntimeError("유효한 라벨이 없습니다. 데이터 라벨을 확인하세요.")

    tp = confusion.diagonal(dim1=1, dim2=2).float()
    denominator = confusion.sum(dim=1) + confusion.sum(dim=2)
    f1 = (2 * tp) / denominator.clamp_min(1)
    return {
        "loss": total_loss_sum / total_valid,
        "macro_f1": f1.mean().item(),
        "f1_per_class": f1.tolist(),
        "confusion_matrix": confusion.tolist(),
        "n_images": total_images,
        "valid_label_count": total_valid,
        "skipped_all_ignored_batches": skipped,
    }


def fit(model, train_loader, val_loader, config: TrainConfig, device):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    run_dir = OUTPUT_ROOT / ("train_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    run_dir.mkdir(parents=True, exist_ok=False)
    best_path = run_dir / "best_model.pth"
    metadata = {
        "architecture": "eva_x_small_patch16", "label_strategy": "u_ignore",
        "class_names": list(CLASS_NAMES), "class_map": {"0": 0, "1": 1, "-1": IGNORE_INDEX},
        "target_labels": list(TARGET_LABELS), "selection_metric": "minimum val loss",
        "pretrained_imagenet": False,
        "pretrained_source": "MapleF/eva_x",
        "pretrained_checkpoint": "eva_x_small_patch16_merged520k_mim.pt",
        "normalization_mean": list(preprocessing_base.MEAN),
        "normalization_std": list(preprocessing_base.STD),
        "csv_names": dict(preprocessing_base.CSV_NAMES),
        "preprocessing": "cxr_preprocessing.py / cxr_preprocessing_uignore.py",
        "config": asdict(config),
    }
    (run_dir / "config.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    history, best_loss, bad_epochs = [], math.inf, 0
    for epoch in range(1, config.epochs + 1):
        train = run_epoch(model, train_loader, device, optimizer)
        val = run_epoch(model, val_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": train["loss"], "val_loss": val["loss"],
            "train_macro_f1": train["macro_f1"], "val_macro_f1": val["macro_f1"],
            "train_valid_labels": train["valid_label_count"],
            "val_valid_labels": val["valid_label_count"],
        })
        with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        print(f"Epoch {epoch}/{config.epochs} | train loss {train['loss']:.4f} | "
              f"val loss {val['loss']:.4f} | val macro F1 {val['macro_f1']:.4f}", flush=True)
        if val["loss"] < best_loss:
            best_loss, bad_epochs = val["loss"], 0
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch, "val_metrics": val, **metadata}, best_path)
            (run_dir / "best_val_metrics.json").write_text(json.dumps(val, indent=2), encoding="utf-8")
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                print("Early stopping", flush=True)
                break
    print("Best checkpoint:", best_path)
    return best_path


def main():
    config = TrainConfig()
    set_seed(config.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU(cuda)가 인식되지 않습니다. 환경을 확인하세요.")
    device = torch.device("cuda")
    print("Device:", device)
    loaders = build_dataloaders(
        csv_dir=CSV_DIR, image_root=IMAGE_ROOT, image_layout="flat",
        image_size=config.image_size, batch_size=config.batch_size,
        num_workers=config.num_workers, seed=config.seed,
    )
    for split, loader in loaders.items():
        print(f"{split}: {len(loader.dataset)} images / {len(loader)} batches")
    for split in ("train", "validate"):
        check_uignore_batch(*next(iter(loaders[split])))
    # Note: test is loaded for split integrity but never used for training or selection.
    model = EVAXSUIgnore().to(device)
    fit(model, loaders["train"], loaders["validate"], config, device)


if __name__ == "__main__":
    main()
