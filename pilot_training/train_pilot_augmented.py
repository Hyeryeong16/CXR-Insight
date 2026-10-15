

from __future__ import annotations
from pathlib import Path
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import DenseNet121_Weights, densenet121

import sys
import torchvision



PROJECT_DIR = Path("/home/team30103/chest-xray-multilabel")
CSV_DIR = PROJECT_DIR / "data" / "processed" / "pilot"

TRAIN_CSV = CSV_DIR / "pilot_train.csv"
VAL_CSV = CSV_DIR / "pilot_val.csv"
TEST_CSV = CSV_DIR / "pilot_test.csv"

OUTPUT_ROOT = PROJECT_DIR / "runs" / "pilot" / "augmented"
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
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = densenet121(weights=weights)
        self.backbone.classifier = nn.Linear(
            self.backbone.classifier.in_features, 9
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

def multiclass_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # 각 소견의 3클래스 CrossEntropy를 동일한 비중으로 평균합니다.
    # 소견 간에 softmax를 취하거나 9개를 하나의 클래스 문제로 취급하지 않습니다.
    return torch.stack([
        nn.functional.cross_entropy(logits[:, i, :], labels[:, i])
        for i in range(3)
    ]).mean()

print("소견별 CrossEntropyLoss 평균 함수 정의 완료")

def run_epoch(model, loader, device, optimizer=None) -> dict:
    training = optimizer is not None
    model.train(training)
    total_loss, n_images = 0.0, 0
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
            loss = multiclass_loss(logits, labels)
            if not torch.isfinite(loss).item():
                raise RuntimeError("손실이 NaN/Inf입니다. 입력과 학습률을 확인하세요.")
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.shape[0]
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
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    run_name = ("smoke_" if smoke_test else "train_") + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path(config.output_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    best_path = output_dir / ("demo_checkpoint.pth" if smoke_test else "best_model.pth")
    metadata = {
        "augmentation": AUGMENTATION_CONFIG,
        "architecture": "densenet121",
        "target_labels": TARGET_LABELS,
        "class_names": CLASS_NAMES,
        "class_map": CLASS_MAP,
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
        })
        print(f"Epoch {epoch}/{config.epochs} | train loss {train['loss']:.4f} | "
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


# 서버의 전처리 모듈과 실제 pilot 데이터를 연결합니다.
sys.path.insert(0, str(PROJECT_DIR))
from cxr_preprocessing import build_dataloaders

if device.type != "cuda":
    raise RuntimeError("GPU가 인식되지 않았습니다. Python 환경을 확인하세요.")

set_seed(42)
loaders = build_dataloaders(
    csv_dir=CSV_DIR,
    image_root=Path("/home/team30103/physionet_data"),
    image_layout="flat",
    image_size=224,
    batch_size=4,
    num_workers=0,
    seed=42,
)
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, RandomAffine

AUGMENTATION_CONFIG = {
    "enabled_splits": ["train"],
    "rotation_degrees": 3.0,
    "translate_fraction": [0.02, 0.02],
    "interpolation": "bilinear",
    "fill_before_normalization": 0.0,
    "horizontal_flip": False,
    "vertical_flip": False,
    "stage": "after resize, before final normalization",
}

class TrainAugmentedDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.mean = torch.tensor(MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(STD, dtype=torch.float32).view(3, 1, 1)
        self.augment = RandomAffine(
            degrees=3.0,
            translate=(0.02, 0.02),
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, labels = self.base_dataset[index]
        # 정규화를 되돌리고 증강한 뒤 같은 기준으로 다시 정규화합니다.
        pixels = (image * self.std + self.mean).clamp(0.0, 1.0)
        pixels = self.augment(pixels)
        return (pixels - self.mean) / self.std, labels

base_train_loader = loaders["train"]
train_loader = DataLoader(
    TrainAugmentedDataset(base_train_loader.dataset),
    batch_size=base_train_loader.batch_size,
    shuffle=True,
    num_workers=base_train_loader.num_workers,
    pin_memory=base_train_loader.pin_memory,
    drop_last=base_train_loader.drop_last,
    generator=base_train_loader.generator,
    collate_fn=base_train_loader.collate_fn,
    worker_init_fn=base_train_loader.worker_init_fn,
)
loaders["train"] = train_loader
val_loader = loaders["validate"]
test_loader = loaders["test"]

assert isinstance(train_loader.dataset, TrainAugmentedDataset)
assert not isinstance(val_loader.dataset, TrainAugmentedDataset)
assert not isinstance(test_loader.dataset, TrainAugmentedDataset)
print("Train 전용 증강: 회전 ±3도 / 가로·세로 이동 최대 2%")
print("Validation과 Test에는 증강을 적용하지 않습니다.")


for name, loader in loaders.items():
    print(f"{name}: {len(loader.dataset)}장 / {len(loader)}배치")

check_batch(*next(iter(train_loader)))
check_batch(*next(iter(val_loader)))
print("실제 데이터 연결 및 배치 검사 완료")



RUN_REAL_TRAINING = True

if RUN_REAL_TRAINING:
    real_config = TrainConfig(
        epochs=10,
        learning_rate=1e-4,
        weight_decay=1e-4, 
	patience=3,
        output_root=str(OUTPUT_ROOT),
    )
    set_seed(real_config.seed)
    print("GPU 시험 학습 시작:", device, flush=True)
    real_model = CXRModel(pretrained=True).to(device)
    _, real_best_path = fit(
        real_model, train_loader, val_loader, real_config, device
    )
    real_best_model, real_checkpoint = load_checkpoint(real_best_path, device)
    print("저장 모델 복원 확인 완료:", real_best_path)
