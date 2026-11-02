
import torch

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def prepare_xrv_images(images):
    """기존 DenseNet 이미지 -> XRV 이미지 변환"""

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"이미지 형태 오류: {images.shape}")

    mean = images.new_tensor(MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(STD).view(1, 3, 1, 1)

    # 1. ImageNet 정규화 해제
    images = images * std + mean

    # 2. RGB 3채널 -> 흑백 1채널
    images = images[:, :1, :, :]

    # 3. [0,1] -> [-1024,1024]
    images = (images * 2.0 - 1.0) * 1024.0

    return images.contiguous()
