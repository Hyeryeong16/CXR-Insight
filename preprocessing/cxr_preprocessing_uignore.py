"""U-Ignore용 MIMIC-CXR DataLoader.

기존 cxr_preprocessing.py의 이미지 전처리 및 데이터 검증을 재사용한다.
불확실 레이블을 2에서 -100으로 변환하여 Loss 계산에서 제외한다.

프로젝트 preprocessing/ 폴더에 배치한다.
"""

from __future__ import annotations

import torch

from preprocessing.cxr_preprocessing import (
    build_dataloaders as _build_multiclass_dataloaders
)

IGNORE_INDEX = -100
TARGET_LABELS = ("Cardiomegaly", "Edema", "Pleural Effusion")
CLASS_NAMES = ("negative", "positive")


def build_dataloaders(
    csv_dir,
    image_root,
    image_layout="mimic",
    image_size=224,
    batch_size=4,
    num_workers=0,
    seed=42,
):
    """원본 DataLoader와 동일한 입력 규격, U-Ignore 라벨 반환.

    반환 labels: [B, 3], torch.long; 0=음성, 1=양성, -100=무시.
    원본 CSV의 -1만 무시 대상으로 변환한다.
    NaN은 원본 전처리와 동일하게 오류로 처리한다.
    """
    loaders = _build_multiclass_dataloaders(
        csv_dir=csv_dir,
        image_root=image_root,
        image_layout=image_layout,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )

    for split, loader in loaders.items():
        dataset = loader.dataset
        # 원본 전처리의 CXRDataset은 labels를 [N, 3] long tensor로 보유.
        labels = dataset.labels
        if labels.ndim != 2 or labels.shape[1] != len(TARGET_LABELS):
            raise ValueError(f"{split}: 예상하지 못한 라벨 형태: {tuple(labels.shape)}")
        if labels.dtype != torch.long or not torch.isin(
            labels, torch.tensor([0, 1, 2], dtype=torch.long)
        ).all().item():
            raise ValueError(f"{split}: 기존 U-MultiClass 라벨은 0/1/2여야 합니다.")

        # 반드시 새 텐서를 만들어 이 DataLoader 내 라벨만 변경.
        dataset.labels = torch.where(
            labels == 2, torch.full_like(labels, IGNORE_INDEX), labels
        )
        count = int((dataset.labels == IGNORE_INDEX).sum().item())
        print(f"{split}: U-Ignore 라벨 준비 완료 (무시 라벨 {count:,}개)")

    return loaders


def check_uignore_batch(images: torch.Tensor, labels: torch.Tensor, image_size=224):
    """U-Ignore DataLoader의 배치 규격 검증."""
    if images.ndim != 4 or tuple(images.shape[1:]) != (3, image_size, image_size):
        raise ValueError(f"이미지 형태 오류: {tuple(images.shape)}")
    if images.dtype != torch.float32 or not torch.isfinite(images).all().item():
        raise ValueError("이미지는 유한한 float32 값이어야 합니다.")
    if labels.shape != (images.shape[0], 3) or labels.dtype != torch.long:
        raise ValueError(f"라벨 형태/자료형 오류: {tuple(labels.shape)}, {labels.dtype}")
    if not torch.isin(labels, torch.tensor([0, 1, IGNORE_INDEX], device=labels.device)).all().item():
        raise ValueError("라벨은 0/1/-100만 허용됩니다.")
