
"""TorchXRayVision DenseNet121 + U-Ignore training."""

from __future__ import annotations

import sys
import csv
import json
import random
import math
from pathlib import Path
from datetime import datetime

import torch
import torchxrayvision as xrv
from torch import nn
from torch.utils.data import DataLoader, Subset

# 프로젝트 경로
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from cxr_preprocessing_uignore import (
    build_dataloaders,
    check_uignore_batch,
    IGNORE_INDEX,
    TARGET_LABELS,
)
from cxr_xrv_adapter import prepare_xrv_images

# 기존 학습 코드의 U-Ignore 평가 함수 재사용
from train_densenet121_uignore_full import run_epoch

CSV_DIR = PROJECT_DIR / "data" / "processed"
IMAGE_ROOT = Path("/home/team30103/physionet_data")
OUTPUT_ROOT = PROJECT_DIR / "torchxrayvision" / "runs"

# True: 소량 시험 학습 / False: 전체 본 학습
PILOT = False

EPOCHS = 2 if PILOT else 10
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 3
SEED = 42


class XRVUIgnore(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = xrv.models.DenseNet(
            weights="densenet121-res224-chex"
        )

        # 기존 18개 소견 분류 헤드 대신
        # 3개 소견 x 2클래스 출력
        in_features = self.backbone.classifier.in_features
        self.classifier = nn.Linear(in_features, 6)

    def forward(self, images):
        images = prepare_xrv_images(images)

        # XRV 사전학습 특징 추출
        features = self.backbone.features2(images)

        # [B, 6] -> [B, 3, 2]
        return self.classifier(features).reshape(-1, 3, 2)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU가 필요합니다.")

    device = torch.device("cuda")
    print("Device:", device)
    print("Mode:", "PILOT" if PILOT else "FULL")

    loaders = build_dataloaders(
        csv_dir=CSV_DIR,
        image_root=IMAGE_ROOT,
        image_layout="flat",
        image_size=224,
        batch_size=BATCH_SIZE,
        num_workers=0,
        seed=SEED,
    )

    train_loader = loaders["train"]
    val_loader = loaders["validate"]

    if PILOT:
        # 전체 데이터에서 고정된 소량 샘플 선택
        rng = random.Random(SEED)

        train_indices = rng.sample(
            range(len(train_loader.dataset)), 128
        )
        val_indices = rng.sample(
            range(len(val_loader.dataset)), 64
        )

        train_loader = DataLoader(
            Subset(train_loader.dataset, train_indices),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            Subset(val_loader.dataset, val_indices),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

    print("Train images:", len(train_loader.dataset))
    print("Val images:", len(val_loader.dataset))

    # 배치 규격 검사
    images, labels = next(iter(train_loader))
    check_uignore_batch(images, labels)

    model = XRVUIgnore().to(device)

    # 출력 확인
    with torch.no_grad():
        output = model(images.to(device))
    print("Model output shape:", output.shape)
    assert output.shape == (len(images), 3, 2)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    run_name = (
        ("pilot_" if PILOT else "full_")
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir = OUTPUT_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "architecture": "XRV DenseNet121",
        "weights": "densenet121-res224-chex",
        "label_strategy": "U-Ignore",
        "target_labels": list(TARGET_LABELS),
        "selection_metric": "minimum validation loss",
        "pilot": PILOT,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "preprocessing": "cxr_xrv_adapter.py",
    }

    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    best_loss = math.inf
    bad_epochs = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train = run_epoch(
            model, train_loader, device, optimizer
        )
        val = run_epoch(model, val_loader, device)

        history.append({
            "epoch": epoch,
            "train_loss": train["loss"],
            "val_loss": val["loss"],
            "train_macro_f1": train["macro_f1"],
            "val_macro_f1": val["macro_f1"],
        })

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train loss {train['loss']:.4f} | "
            f"val loss {val['loss']:.4f} | "
            f"val macro F1 {val['macro_f1']:.4f}",
            flush=True,
        )

        with (run_dir / "history.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=history[0].keys()
            )
            writer.writeheader()
            writer.writerows(history)

        if val["loss"] < best_loss:
            best_loss = val["loss"]
            bad_epochs = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_metrics": val,
                "config": config,
            }, run_dir / "best_model.pth")

            print("Best model saved!", flush=True)
        else:
            bad_epochs += 1

            if bad_epochs >= PATIENCE:
                print("Early stopping")
                break

    print("Training finished!")
    print("Results:", run_dir)


if __name__ == "__main__":
    main()
