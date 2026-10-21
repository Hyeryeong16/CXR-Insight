
import sys
from pathlib import Path

# 프로젝트 최상위 경로
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torchxrayvision as xrv

from cxr_preprocessing import CSV_DIR, IMAGE_ROOT, IMAGE_LAYOUT
from cxr_preprocessing_uignore import (
    build_dataloaders,
    check_uignore_batch,
)
from cxr_xrv_adapter import prepare_xrv_images

# 1. 기존 U-Ignore DataLoader 불러오기
loaders = build_dataloaders(
    csv_dir=CSV_DIR,
    image_root=IMAGE_ROOT,
    image_layout=IMAGE_LAYOUT,
    image_size=224,
    batch_size=4,
    num_workers=0,
    seed=42,
)

# 2. Train 데이터에서 한 배치 가져오기
images, labels = next(iter(loaders["train"]))
check_uignore_batch(images, labels)

# 3. XRV 입력 규격으로 변환
xrv_images = prepare_xrv_images(images)

print("기존 이미지:", images.shape)
print("XRV 이미지:", xrv_images.shape)
print("레이블:", labels.shape)
print("XRV 픽셀 범위:",
      float(xrv_images.min()),
      float(xrv_images.max()))
print("레이블 값:", torch.unique(labels).tolist())

# 4. GPU에서 XRV 모델 추론 테스트
device = "cuda" if torch.cuda.is_available() else "cpu"
model = xrv.models.DenseNet(
    weights="densenet121-res224-chex"
).to(device)
model.eval()

with torch.no_grad():
    outputs = model(xrv_images.to(device))

print("모델 출력:", outputs.shape)
print("XRV 데이터 연결 및 Forward 성공!")
