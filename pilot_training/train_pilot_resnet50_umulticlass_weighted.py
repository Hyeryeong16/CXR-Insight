"""기존 train_pilot.py의 학습 조건을 유지한 ResNet50 클래스 가중치 pilot 비교 학습.

서버에서 실행:
    python /home/team30103/chest-xray-multilabel/pilot_training/train_pilot_resnet50_weighted.py

상단 PROJECT_DIR / CSV_DIR / IMAGE_ROOT만 실제 경로에 맞추세요.
전처리 모듈은 프로젝트 루트의 cxr_preprocessing.py를 그대로 사용합니다.
ImageNet IMAGENET1K_V1을 사용하며 최초 실행 시 가중치 다운로드가 필요합니다.
기본 ResNet50과 같은 CSV를 사용해야 비교할 수 있습니다.
train 클래스 빈도의 역제곱근 가중치를 소견별로 계산합니다.
학습 손실에만 가중치를 적용하고 validation loss는 기존 무가중 CE를 유지합니다.
train weighted loss와 train unweighted CE를 모두 기록합니다.
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

OUTPUT_ROOT = PROJECT_DIR / "runs" / "pilot_resnet50_weighted"
IMAGE_ROOT = Path("/home/team30103/physionet_data")
print("Python:", sys.version.split()[0])
print("프로젝트 인터프리터:", sys.executable)
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("GPU 사용 가능:", torch.cuda.is_available())
print("결과 저장 루트:", OUTPUT_ROOT)

TARGET_LABELS = ["Cardiomegaly", "Edema", "Pleural Effusion"]
CLASS_NAMES = ["negative", "positive", "uncertain"]
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
            self.backbone.fc.in_features, 9
        )
        self.initialized_with_imagenet = pretrained

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # [B, 9] -> [B, 소견 3개, 클래스 3개]
        # 손실 함수 입력에는 softmax를 적용하지 않습니다.
        return self.backbone(images).reshape(-1, 3, 3)

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

def weights_from_counts(counts):
    """각 소견에서 w=1/sqrt(n), 관측된 클래스 가중치의 평균이 1이 되게 정규화."""
    if len(counts) != 3:
        raise ValueError("소견별 3행의 클래스 수가 필요합니다.")
    result = []
    for row in counts:
        if len(row) != 3 or any(n < 0 or int(n) != n for n in row) or sum(row) == 0:
            raise ValueError("클래스 수는 0 이상의 정수 3개이며 소견별 합이 양수여야 합니다.")
        raw = [1.0 / math.sqrt(n) if n else 0.0 for n in row]
        mean = sum(raw) / sum(n > 0 for n in row)
        result.append([value / mean for value in raw])
    return result


def build_class_weights(train_dataset):
    # 실제 train Dataset의 변환된 라벨(0/1/2)만 사용합니다.
    # DataLoader를 순회하지 않으므로 shuffle 난수 상태를 바꾸지 않습니다.
    if not hasattr(train_dataset, "labels"):
        raise ValueError("train Dataset에 labels가 없습니다. CXRDataset.labels를 확인하세요.")
    labels = torch.as_tensor(train_dataset.labels)
    if labels.dtype != torch.long or labels.ndim != 2 or labels.shape[1] != 3:
        raise ValueError("train labels는 [N, 3] 형태의 torch.long이어야 합니다.")
    if len(labels) != len(train_dataset) or not len(labels):
        raise ValueError("train labels 개수가 Dataset과 다르거나 비어 있습니다.")
    if not ((labels >= 0) & (labels <= 2)).all().item():
        raise ValueError("train labels는 매핑 후 0/1/2만 허용합니다.")
    counts = [torch.bincount(labels[:, i], minlength=3).tolist() for i in range(3)]
    values = weights_from_counts(counts)
    print("\nTrain 전용 클래스 가중치: 음성 / 양성 / 불확실")
    for label, row, weights in zip(TARGET_LABELS, counts, values):
        print(f"{label}: counts={row}, weights={[round(w, 4) for w in weights]}")
        for class_name, count in zip(CLASS_NAMES, row):
            if count == 0:
                print(f"주의: train에 {label}/{class_name}이 없어 가중치는 0입니다. "
                      "가중치만으로 없는 클래스를 학습할 수는 없습니다.")
    return torch.tensor(values, dtype=torch.float32), counts


def multiclass_loss(logits: torch.Tensor, labels: torch.Tensor,
                    class_weights=None) -> torch.Tensor:
    # 각 소견은 별도의 음성/양성/불확실 3클래스 분류입니다.
    return torch.stack([
        nn.functional.cross_entropy(
            logits[:, i, :], labels[:, i],
            weight=None if class_weights is None else class_weights[i],
        )
        for i in range(3)
    ]).mean()

print("소견별 CrossEntropyLoss 평균 함수 정의 완료")

def run_epoch(model, loader, device, optimizer=None, class_weights=None) -> dict:
    training = optimizer is not None
    model.train(training)
    total_loss, total_unweighted_loss, n_images = 0.0, 0.0, 0
    # 소견별 [정답 클래스, 예측 클래스] confusion matrix
    confusion = torch.zeros(3, 3, 3, dtype=torch.long)
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            check_batch(images, labels)
            images = images.to(device)
            labels = labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            # 검증 시에는 가중치를 넘겨도 무가중 손실로 계산합니다.
            loss = multiclass_loss(logits, labels, class_weights if training else None)
            unweighted = multiclass_loss(logits.detach(), labels) if training else loss
            if not torch.isfinite(loss).item():
                raise RuntimeError("손실이 NaN/Inf입니다. 입력과 학습률을 확인하세요.")
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.shape[0]
            total_unweighted_loss += unweighted.item() * images.shape[0]
            n_images += images.shape[0]
            predictions = logits.detach().argmax(dim=-1).cpu()
            targets = labels.detach().cpu()
            for i in range(3):
                indices = targets[:, i] * 3 + predictions[:, i]
                confusion[i] += torch.bincount(indices, minlength=9).reshape(3, 3)
    if n_images == 0:
        raise ValueError("DataLoader에서 반환된 이미지가 없습니다.")
    tp = confusion.diagonal(dim1=1, dim2=2).float()
    denominators = confusion.sum(dim=1) + confusion.sum(dim=2)
    # 관측·예측이 모두 없는 클래스의 F1은 0으로 기록합니다.
    f1 = 2 * tp / denominators.clamp_min(1)
    return {
        "loss": total_loss / n_images,
        "unweighted_loss": total_unweighted_loss / n_images,
        "macro_f1": f1.mean().item(),
        "f1_per_class": f1.tolist(),
        "confusion_matrix": confusion.tolist(),
        "n_images": n_images,
    }

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
    class_weights, class_counts = build_class_weights(train_loader.dataset)
    class_weights = class_weights.to(device)
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
        "input": {
            "channels": 3, "height": IMAGE_SIZE, "width": IMAGE_SIZE,
            "scale": "uint8 / 255", "mean": MEAN, "std": STD,
            "geometry": "preserve aspect ratio, pad square, resize",
        },
        "initialized_with_imagenet": model.initialized_with_imagenet,
        "selection_metric": "minimum unweighted val loss",
        "class_weighting": {
            "method": "inverse_sqrt_frequency",
            "normalization": "per finding, mean weight of observed classes = 1",
            "source": "train Dataset labels only",
            "counts": class_counts,
            "weights": class_weights.detach().cpu().tolist(),
            "absent_train_class_weight": 0,
            "applied_to": "training loss only",
            "weighted_ce_reduction": "PyTorch weighted mean per batch per finding",
        },
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
        train = run_epoch(model, train_loader, device, optimizer, class_weights)
        val = run_epoch(model, val_loader, device)
        history.append({
            "epoch": epoch, "train_loss": train["loss"], "val_loss": val["loss"],
            "train_unweighted_loss": train["unweighted_loss"],
            "train_macro_f1": train["macro_f1"], "val_macro_f1": val["macro_f1"],
        })
        print(f"Epoch {epoch}/{config.epochs} | train weighted loss {train['loss']:.4f} | "
              f"train CE {train['unweighted_loss']:.4f} | "
              f"val loss {val['loss']:.4f} | val macro F1 {val['macro_f1']:.4f}")
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
    model = CXRModel(pretrained=False)  # 가중치는 파일에서 복원하므로 다운로드 불필요
    model.load_state_dict(checkpoint["model_state_dict"])
    model.initialized_with_imagenet = checkpoint["initialized_with_imagenet"]
    model.to(device).eval()
    return model, checkpoint

@torch.inference_mode()
def predict_probabilities(model, images):
    """반환 [B, 3, 3]. 소견별 음성·양성·불확실 확률입니다."""
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
    """학습 조건을 바꾸지 않고 best 모델의 클래스별 결과를 표시합니다."""
    print("\nBest validation: loss={:.4f}, macro F1={:.4f}".format(
        metrics["loss"], metrics["macro_f1"]
    ))
    print("소견 | 클래스 | 정답 수 | 예측 수 | Recall | F1")
    for label, matrix, f1_values in zip(
        TARGET_LABELS, metrics["confusion_matrix"], metrics["f1_per_class"]
    ):
        for cls, class_name in enumerate(CLASS_NAMES):
            support = sum(matrix[cls])
            predicted = sum(row[cls] for row in matrix)
            recall = f"{matrix[cls][cls] / support:.4f}" if support else "N/A"
            print(f"{label} | {class_name} | {support} | {predicted} | "
                  f"{recall} | {f1_values[cls]:.4f}")
    print("정답 수 0인 클래스는 Recall=N/A이며 기존 규칙대로 F1은 0으로 집계됩니다.")


def main():
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

    real_config = TrainConfig(
        epochs=10,
        learning_rate=1e-4,
        weight_decay=1e-4,
        patience=3,
        output_root=str(OUTPUT_ROOT),
    )
    set_seed(real_config.seed)
    print("ResNet50 클래스 가중치 GPU 시험 학습 시작:", device, flush=True)
    print("학습만 weighted CE / 검증·best 선택은 기존과 같은 unweighted CE")
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
    main()
