"""기존 train_pilot.py의 학습 조건을 유지한 ResNet50 + U-ignore pilot 학습.

서버에서 실행:
    python /home/team30103/chest-xray-multilabel/pilot_training/train_pilot_resnet50_uignore.py

상단 PROJECT_DIR / CSV_DIR / IMAGE_ROOT만 실제 경로에 맞추세요.
전처리 모듈은 프로젝트 루트의 cxr_preprocessing.py를 그대로 사용합니다.
ImageNet IMAGENET1K_V1을 사용하며 최초 실행 시 가중치 다운로드가 필요합니다.
기존 pilot CSV와 전처리 모듈은 그대로 사용합니다.
Dataset의 불확실 라벨 2(원본 -1)를 소견별로 손실·지표에서 제외합니다.
이미지 전체를 제거하거나 불확실을 음성으로 바꾸지 않습니다.
출력은 [B, 3, 2]: 음성/양성만 예측합니다. 클래스 가중치는 없습니다.
기존 3클래스 loss 및 Macro F1과 수치를 직접 비교하지 마세요.
실행 시 먼저 작은 CPU 텐서로 제외 마스크·역전파 자체 검사를 합니다.
--self-test 옵션으로 자체 검사만 실행할 수 있습니다.
GPU가 없으면 실제 학습은 시작하지 않습니다. test 평가는 실행하지 않습니다.
"""

from __future__ import annotations
import csv
import json
import math
import random
import hashlib
import shutil
from tempfile import TemporaryDirectory
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

import sys
import torchvision



PROJECT_DIR = Path("/home/team30103/chest-xray-multilabel")
CSV_DIR = PROJECT_DIR / "data" / "processed" / "pilot"

TRAIN_CSV = CSV_DIR / "pilot_train.csv"
VAL_CSV = CSV_DIR / "pilot_val.csv"
TEST_CSV = CSV_DIR / "pilot_test.csv"

OUTPUT_ROOT = PROJECT_DIR / "runs" / "pilot_resnet50_uignore"
IMAGE_ROOT = Path("/home/team30103/physionet_data")
print("Python:", sys.version.split()[0])
print("프로젝트 인터프리터:", sys.executable)
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("GPU 사용 가능:", torch.cuda.is_available())
print("결과 저장 루트:", OUTPUT_ROOT)

TARGET_LABELS = ["Cardiomegaly", "Edema", "Pleural Effusion"]
CLASS_NAMES = ["negative", "positive"]
IGNORE_INDEX = 2  # 기존 Dataset에서 원본 -1이 변환된 값
CLASS_MAP = {0: 0, 1: 1, -1: 2}
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


@dataclass
class TrainConfig:
    # 시작 설정입니다. 최적값이 아니며 검증 결과와 GPU 환경에 따라 조정합니다.
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 3
    seed: int = 42
    output_root: str = str(OUTPUT_ROOT)

def set_seed(seed: int = 42) -> None:
    """실제 학습에서는 DataLoader와 모델을 만들기 전에 호출하세요."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 같은 환경에서 재현성을 높이는 설정. 모든 환경의 완전한 일치는 보장하지 않습니다.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


set_seed(42)
device = choose_device()
print("사용 장치:", device)

if device.type == "cpu":
    torch.set_num_threads(min(4, torch.get_num_threads()))
print("장치:", device)
print("소견 순서:", TARGET_LABELS)
print("클래스 순서:", CLASS_NAMES)
print("원본 → 학습 라벨:", CLASS_MAP)

class CXRModel(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        # True: ImageNet 가중치 사용 (최초 다운로드 필요).
        # False: 무작위 초기화. CSV 없는 동작 확인에서만 사용합니다.
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet50(weights=weights)
        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features, 6
        )
        self.initialized_with_imagenet = pretrained

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # [B, 6] -> [B, 소견 3개, 음성/양성 2개]
        # 손실 함수 입력에는 softmax를 적용하지 않습니다.
        return self.backbone(images).reshape(-1, 3, 2)

def check_batch(images: torch.Tensor, labels: torch.Tensor) -> None:
    if images.ndim != 4 or tuple(images.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"이미지는 [B, 3, 224, 224]여야 합니다: {images.shape}")
    if images.dtype != torch.float32 or not torch.isfinite(images).all().item():
        raise ValueError("이미지는 유한한 float32 값이어야 합니다.")
    if labels.shape != (images.shape[0], 3) or labels.dtype != torch.long:
        raise ValueError("정답은 [B, 3] 형태의 torch.long이어야 합니다.")
    if not ((labels >= 0) & (labels <= 2)).all().item():
        raise ValueError("정답은 0/1/2만 허용합니다. Dataset에서 -1을 2로 바꾸세요.")

print("모델 클래스와 배치 검사 함수 정의 완료")

def masked_loss_parts(logits, labels):
    """[B, 3, 2] logits에서 소견별 확정 라벨 CE 합과 유효 라벨 수 반환."""
    if logits.shape != (*labels.shape, 2):
        raise ValueError("U-ignore logits는 [B, 3, 2]여야 합니다.")
    losses = nn.functional.cross_entropy(
        logits.reshape(-1, 2), labels.reshape(-1),
        ignore_index=IGNORE_INDEX, reduction="none",
    ).reshape_as(labels)
    valid = labels.ne(IGNORE_INDEX)
    return losses.sum(dim=0), valid.sum(dim=0)


def masked_loss(logits, labels):
    sums, counts = masked_loss_parts(logits, labels)
    present = counts.gt(0)
    if not present.any().item():
        # 모두 불확실해도 NaN 없이 역전파 가능한 0. 실제 epoch에서는 이 배치를 건너뜁니다.
        return logits.sum() * 0.0
    return (sums[present] / counts[present]).mean()


def metrics_from_confusion(matrices):
    """소견별 2x2 혼동행렬. 행=정답, 열=예측. 불확실 정답은 집계 전 제외."""
    if len(matrices) != 3:
        raise ValueError("세 소견의 혼동행렬이 필요합니다.")
    f1_all, recalls, precisions, positive_f1 = [], [], [], []
    for matrix in matrices:
        if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
            raise ValueError("U-ignore 혼동행렬은 소견마다 2x2여야 합니다.")
        f1, recall, precision = [], [], []
        for cls in range(2):
            tp = matrix[cls][cls]
            support = sum(matrix[cls])
            predicted = sum(row[cls] for row in matrix)
            f1.append(2 * tp / (support + predicted) if support + predicted else 0.0)
            recall.append(tp / support if support else None)
            precision.append(tp / predicted if predicted else None)
        f1_all.append(f1)
        recalls.append(recall)
        precisions.append(precision)
        positive_f1.append(f1[1])
    return {
        "macro_f1": sum(sum(row) for row in f1_all) / 6,
        "macro_f1_definition": "mean of 3 findings x 2 classes; true uncertain excluded",
        "mean_positive_f1": sum(positive_f1) / 3,
        "f1_per_class": f1_all,
        "recall_per_class": recalls,
        "precision_per_class": precisions,
        "confusion_matrix": matrices,
    }


def run_epoch(model, loader, device, optimizer=None) -> dict:
    training = optimizer is not None
    model.train(training)
    sums_all = torch.zeros(3, dtype=torch.float64)
    counts_all = torch.zeros(3, dtype=torch.long)
    ignored_all = torch.zeros(3, dtype=torch.long)
    confusion = torch.zeros(3, 2, 2, dtype=torch.long)
    n_images, skipped_batches = 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            check_batch(images, labels)
            n_images += images.shape[0]
            ignored_all += labels.eq(IGNORE_INDEX).sum(dim=0).cpu()
            # 확정 라벨이 전혀 없는 배치는 forward와 AdamW 업데이트를 모두 건너뜁니다.
            if not labels.ne(IGNORE_INDEX).any().item():
                skipped_batches += 1
                continue
            images = images.to(device)
            labels = labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            sums, counts = masked_loss_parts(logits, labels)
            present = counts.gt(0)
            loss = (sums[present] / counts[present]).mean()
            if not torch.isfinite(loss).item():
                raise RuntimeError("손실이 NaN/Inf입니다. 입력과 학습률을 확인하세요.")
            if training:
                loss.backward()
                optimizer.step()
            sums_all += sums.detach().cpu().double()
            counts_all += counts.detach().cpu()
            predictions = logits.detach().argmax(dim=-1).cpu()
            targets = labels.detach().cpu()
            for i in range(3):
                valid = targets[:, i].ne(IGNORE_INDEX)
                indices = targets[valid, i] * 2 + predictions[valid, i]
                confusion[i] += torch.bincount(indices, minlength=4).reshape(2, 2)
    if n_images == 0:
        raise ValueError("DataLoader에서 반환된 이미지가 없습니다.")
    present = counts_all.gt(0)
    if not present.any().item():
        raise ValueError("이 split의 모든 라벨이 불확실합니다. 학습/검증할 확정 라벨이 없습니다.")
    # 배치별 평균을 다시 단순 평균하지 않습니다. split 전체 유효 수로 소견별 평균을 계산합니다.
    epoch_loss = (sums_all[present] / counts_all[present]).mean().item()
    per_finding = [
        sums_all[i].item() / counts_all[i].item() if counts_all[i].item() else None
        for i in range(3)
    ]
    return {
        "loss": epoch_loss,
        "loss_per_finding": per_finding,
        "n_images": n_images,
        "n_valid_per_finding": counts_all.tolist(),
        "n_ignored_per_finding": ignored_all.tolist(),
        "skipped_all_uncertain_batches": skipped_batches,
        **metrics_from_confusion(confusion.tolist()),
    }


def self_test_uignore():
    """이미지/가중치 다운로드 없이 작은 CPU 텐서로 손실과 gradient 마스크 검사."""
    logits = torch.zeros(2, 3, 2, requires_grad=True)
    labels = torch.tensor([[0, 2, 1], [1, 2, 2]], dtype=torch.long)
    loss = masked_loss(logits, labels)
    torch.testing.assert_close(loss, torch.tensor(math.log(2.0)))
    loss.backward()
    assert torch.equal(logits.grad[labels.eq(2)], torch.zeros_like(logits.grad[labels.eq(2)]))
    assert logits.grad[labels.ne(2)].abs().sum().item() > 0
    changed = logits.detach().clone()
    changed[labels.eq(2)] = torch.tensor([100.0, -100.0])
    torch.testing.assert_close(masked_loss(changed, labels), loss.detach())
    empty_logits = torch.zeros(2, 3, 2, requires_grad=True)
    empty_loss = masked_loss(empty_logits, torch.full_like(labels, 2))
    assert torch.isfinite(empty_loss).item() and empty_loss.item() == 0.0
    empty_loss.backward()
    assert torch.equal(empty_logits.grad, torch.zeros_like(empty_logits))
    # 확정 라벨이 두 소견에만 있는 배치: 분모는 유효 라벨 수여야 합니다.
    _, counts = masked_loss_parts(logits.detach(), labels)
    assert counts.tolist() == [2, 0, 1]
    perfect = metrics_from_confusion([[[2, 0], [0, 1]]] * 3)
    assert perfect["macro_f1"] == 1.0 and perfect["mean_positive_f1"] == 1.0
    print("U-ignore 자체 검사 통과: 불확실 손실·gradient 제외, 유효 분모, 전부 제외된 배치 처리")


def fit(model, train_loader, val_loader, config=None, device=None, *, smoke_test=False):
    """test_loader는 받지 않습니다. 모델 선택은 val loss로만 합니다.

    반환: (학습된 model, best checkpoint의 Path)
    반환 model은 마지막 epoch 상태이므로 평가·예측 전에 best를 다시 불러오세요.
    """
    config = config or TrainConfig()
    if config.epochs < 1 or config.patience < 1:
        raise ValueError("epochs와 patience는 1 이상이어야 합니다.")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("learning_rate는 양의 유한한 값이어야 합니다.")
    if not math.isfinite(config.weight_decay) or config.weight_decay < 0:
        raise ValueError("weight_decay는 0 이상의 유한한 값이어야 합니다.")
    device = torch.device(device) if device is not None else choose_device()
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    run_name = ("smoke_" if smoke_test else "train_") + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path(config.output_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    best_path = output_dir / ("demo_checkpoint.pth" if smoke_test else "best_model.pth")
    metadata = {
        "architecture": "resnet50",
        "pretrained_weights": "IMAGENET1K_V1" if model.initialized_with_imagenet else None,
        "target_labels": TARGET_LABELS,
        "class_names": CLASS_NAMES,
        "class_map": CLASS_MAP,
        "uncertainty_policy": "U-ignore",
        "ignore_index": IGNORE_INDEX,
        "num_classes_per_finding": 2,
        "class_weights": None,
        "loss_reduction": "per-finding CE mean on known labels, then mean over supported findings",
        "evaluation_policy": "exclude true uncertain per finding; macro F1 averages six class scores",
        "input": {
            "channels": 3, "height": IMAGE_SIZE, "width": IMAGE_SIZE,
            "scale": "uint8 / 255", "mean": MEAN, "std": STD,
            "geometry": "preserve aspect ratio, pad square, resize",
        },
        "initialized_with_imagenet": model.initialized_with_imagenet,
        "selection_metric": "minimum val loss",
        "smoke_test": smoke_test,
        "config": asdict(config),
        "torch_version": str(torch.__version__),
    }
    # 실제 Dataset이 이 입력 규격을 적용하는지는 담당자와 확인해야 합니다.
    (output_dir / "config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    history, best_loss, bad_epochs = [], float("inf"), 0
    for epoch in range(1, config.epochs + 1):
        train = run_epoch(model, train_loader, device, optimizer)
        val = run_epoch(model, val_loader, device)
        history.append({
            "epoch": epoch, "train_loss": train["loss"], "val_loss": val["loss"],
            "train_macro_f1": train["macro_f1"], "val_macro_f1": val["macro_f1"],
            "val_mean_positive_f1": val["mean_positive_f1"],
        })
        print(f"Epoch {epoch}/{config.epochs} | train loss {train['loss']:.4f} | "
              f"val loss {val['loss']:.4f} | val macro F1(2-class) {val['macro_f1']:.4f} | "
              f"val mean positive F1 {val['mean_positive_f1']:.4f}")
        with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        if val["loss"] < best_loss:
            best_loss, bad_epochs = val["loss"], 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch, "val_metrics": val, **metadata,
            }, best_path)
            (output_dir / "best_val_metrics.json").write_text(
                json.dumps(val, indent=2), encoding="utf-8"
            )
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                print(f"검증 손실 개선이 {config.patience}회 없어 조기 종료합니다.")
                break
    print(f"저장 위치: {best_path}")
    return model, best_path

print("학습·검증·저장 함수 정의 완료")

def load_checkpoint(path, device=None, *, allow_demo=False):
    """이 코드에서 만든 checkpoint를 불러옵니다. 임시 모델은 기본적으로 차단합니다."""
    device = torch.device(device) if device is not None else choose_device()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("smoke_test", False) and not allow_demo:
        raise ValueError("임시 데이터 모델입니다. 실제 이미지 예측용으로 사용하지 마세요.")
    if checkpoint["target_labels"] != TARGET_LABELS or checkpoint["class_names"] != CLASS_NAMES:
        raise ValueError("checkpoint의 소견·클래스 순서가 현재 코드와 다릅니다.")
    if checkpoint.get("architecture") != "resnet50":
        raise ValueError("ResNet50 checkpoint가 아닙니다. DenseNet 결과 경로와 구분하세요.")
    if checkpoint.get("uncertainty_policy") != "U-ignore" or checkpoint.get("num_classes_per_finding") != 2:
        raise ValueError("U-ignore 2클래스 checkpoint가 아닙니다. 기존 3클래스 모델과 구분하세요.")
    model = CXRModel(pretrained=False)  # 가중치는 파일에서 복원하므로 다운로드 불필요
    model.load_state_dict(checkpoint["model_state_dict"])
    model.initialized_with_imagenet = checkpoint["initialized_with_imagenet"]
    model.to(device).eval()
    return model, checkpoint

@torch.inference_mode()
def predict_probabilities(model, images):
    """반환 [B, 3, 2]. 소견별 음성/양성 확률이며 불확실 확률은 출력하지 않습니다."""
    model.eval()
    device = next(model.parameters()).device
    return model(images.to(device)).softmax(dim=-1).cpu()

print("모델 복원·확률 계산 함수 정의 완료")


def build_pilot_loaders():
    """같은 CSV 내용을 유지하면서 전처리 모듈의 파일명 규칙에 연결합니다."""
    sys.path.insert(0, str(PROJECT_DIR))
    import cxr_preprocessing as preprocessing

    filenames = {
        "train": ("pilot_train.csv", "train.csv"),
        "validate": ("pilot_val.csv", "val.csv"),
        "test": ("pilot_test.csv", "test.csv"),
    }
    sources = {}
    manifest = {}
    for split, names in filenames.items():
        candidates = [CSV_DIR / name for name in names if (CSV_DIR / name).is_file()]
        if not candidates:
            raise FileNotFoundError(f"{CSV_DIR}: {names} 중 해당 split CSV가 필요합니다.")
        if len(candidates) > 1 and candidates[0].read_bytes() != candidates[1].read_bytes():
            raise ValueError(
                f"{split}: {names} 두 파일의 내용이 다릅니다. "
                "기존 DenseNet 학습에 사용한 CSV만 이 폴더에 두세요."
            )
        source = candidates[0]
        sources[split] = source
        manifest[split] = {
            "source_csv": str(source.resolve()),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
        print(f"{split} CSV: {source}")

    # 일부 서버 전처리 모듈은 train.csv, 일부는 pilot_train.csv를 기대합니다.
    # 임시 폴더에 바이트가 같은 복사본을 만들며 원본 파일은 수정하지 않습니다.
    with TemporaryDirectory(prefix="cxr_resnet50_pilot_") as temporary:
        alias_dir = Path(temporary)
        for split, names in filenames.items():
            for name in names:
                shutil.copyfile(sources[split], alias_dir / name)
        loaders = preprocessing.build_dataloaders(
            csv_dir=alias_dir,
            image_root=IMAGE_ROOT,
            image_layout="flat",
            image_size=224,
            batch_size=4,
            num_workers=0,
            seed=42,
        )
    # 기존 build_dataloaders는 생성 시 CSV를 읽고 이미지 절대경로를 보관합니다.
    return loaders, manifest


def report_best_validation(metrics):
    print("\nBest U-ignore validation: loss={:.4f}, macro F1(2-class)={:.4f}, "
          "mean positive F1={:.4f}".format(
              metrics["loss"], metrics["macro_f1"], metrics["mean_positive_f1"]
          ))
    print("소견 | 확정 라벨 평가 수 | 제외한 불확실 정답 수")
    for label, valid, ignored in zip(TARGET_LABELS, metrics["n_valid_per_finding"],
                                     metrics["n_ignored_per_finding"]):
        print(f"{label} | {valid} | {ignored}")
    print("소견 | 클래스 | 정답 수 | 예측 수 | Precision | Recall | F1")
    def fmt(value):
        return f"{value:.4f}" if value is not None else "N/A"
    for i, label in enumerate(TARGET_LABELS):
        matrix = metrics["confusion_matrix"][i]
        for cls, class_name in enumerate(CLASS_NAMES):
            support = sum(matrix[cls])
            predicted = sum(row[cls] for row in matrix)
            precision = metrics["precision_per_class"][i][cls]
            recall = metrics["recall_per_class"][i][cls]
            f1 = metrics["f1_per_class"][i][cls]
            print(f"{label} | {class_name} | {support} | {predicted} | "
                  f"{fmt(precision)} | {fmt(recall)} | {f1:.4f}")
    print("불확실 정답은 해당 소견의 손실·평가에서만 제외했습니다. 이미지는 유지했습니다.")
    print("기존 3클래스 loss/Macro F1과 직접 비교하지 말고 동일 확정 라벨로 재평가하세요.")


def main():
    self_test_uignore()
    if device.type != "cuda":
        raise RuntimeError("GPU가 인식되지 않았습니다. 기존 학습용 Python 환경을 확인하세요.")

    set_seed(42)
    loaders, manifest = build_pilot_loaders()
    train_loader = loaders["train"]
    val_loader = loaders["validate"]
    # test loader의 데이터는 학습·모델 선택·평가에 사용하지 않습니다.
    for name, loader in loaders.items():
        print(f"{name}: {len(loader.dataset)}장 / {len(loader)}배치")
    total = sum(len(loader.dataset) for loader in loaders.values())
    if total != 1000:
        raise ValueError(f"pilot 전체가 {total}장입니다. 기존 1,000장 CSV를 확인하세요.")

    check_batch(*next(iter(train_loader)))
    check_batch(*next(iter(val_loader)))
    print("실제 데이터 연결 및 배치 검사 완료")
    print("U-ignore: Dataset의 2(원본 -1)는 소견별 손실·평가에서 제외; 출력은 음성/양성")
    print("기존 3클래스 loss/Macro F1과 직접 비교할 수 없습니다.")

    real_config = TrainConfig(
        epochs=10,
        learning_rate=1e-4,
        weight_decay=1e-4,
        patience=3,
        output_root=str(OUTPUT_ROOT),
    )
    set_seed(real_config.seed)
    print("ResNet50 + U-ignore GPU 시험 학습 시작:", device, flush=True)
    real_model = CXRModel(pretrained=True).to(device)
    _, real_best_path = fit(
        real_model, train_loader, val_loader, real_config, device
    )
    _, real_checkpoint = load_checkpoint(real_best_path, device)
    (real_best_path.parent / "pilot_csv_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("저장 모델 복원 확인 완료:", real_best_path)
    report_best_validation(real_checkpoint["val_metrics"])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ResNet50 + U-ignore pilot 학습")
    parser.add_argument("--self-test", action="store_true", help="이미지 없이 CPU 손실 마스크 검사만 실행")
    args = parser.parse_args()
    if args.self_test:
        self_test_uignore()
    else:
        main()
