from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image
from torch import nn
from torchvision.models import densenet121

from cxr_preprocessing import preprocess_image, MEAN, STD


# ============================================================
# 사용자 설정
# 파일 위치: 프로젝트/test_hr/cxr_inference_positive_gradcam.py
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

# 1. 사용할 모델
MODEL_PATH = (
    PROJECT_DIR / "runs" / "pilot" / "frozen"
    / "train_20261015_104401_031713"
    / "best_model.pth"
)

# 2. 이미지 폴더
IMAGE_ROOT = Path("/home/team30103/physionet_data")

# 3. 테스트 CSV
TEST_CSV = (
    PROJECT_DIR / "data" / "processed" / "pilot"
    / "pilot_test.csv"
)

# 4. 확인할 이미지 번호: 0부터 시작
IMAGE_INDEX = 4

# 5. 결과 저장 위치
OUTPUT_ROOT = PROJECT_DIR / "test_hr" / "results"


# ============================================================
# 소견·클래스 순서
# ============================================================

FINDING_NAMES = [
    "Cardiomegaly",
    "Edema",
    "Pleural Effusion",
]
CLASS_NAMES = ["Negative", "Positive", "Uncertain"]


# ============================================================
# 모델 정의
# ============================================================

class CXRModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = densenet121(weights=None)
        self.backbone.classifier = nn.Linear(
            self.backbone.classifier.in_features, 9
        )

    def forward(self, images):
        features = self.backbone.features(images)
        features = F.relu(features, inplace=False)

        pooled = F.adaptive_avg_pool2d(features, (1, 1))
        flattened = torch.flatten(pooled, 1)
        logits = self.backbone.classifier(flattened)

        return logits.reshape(-1, 3, 3)


# ============================================================
# Grad-CAM
# ============================================================

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None

        self.handle = (
            model.backbone.features.norm5.register_forward_hook(
                self._capture
            )
        )

    def _capture(self, module, inputs, output):
        self.activations = output.detach().clone()

        if output.requires_grad:
            output.register_hook(self._capture_gradient)

    def _capture_gradient(self, gradient):
        self.gradients = gradient.detach().clone()

    def generate(self, image, finding_idx, class_idx=1):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        self.activations = None
        self.gradients = None

        device = next(self.model.parameters()).device

        with torch.enable_grad():
            inputs = (
                image.detach().clone()
                .unsqueeze(0)
                .to(device)
                .requires_grad_(True)
            )

            logits = self.model(inputs)
            score = logits[0, finding_idx, class_idx]
            score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "특징 지도 또는 기울기를 얻지 못했습니다."
            )

        weights = self.gradients.mean(
            dim=(2, 3), keepdim=True
        )

        cam = (
            weights * self.activations
        ).sum(dim=1, keepdim=True)

        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0].detach().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        self.model.zero_grad(set_to_none=True)

        return cam

    def close(self):
        self.handle.remove()


# ============================================================
# 이미지 한 장의 예측·Grad-CAM 반환
# ============================================================

class CXRInference:
    def __init__(self, model_path, device=None):
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model_path = Path(model_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"모델 파일이 없습니다: {self.model_path}"
            )

        checkpoint = torch.load(
            self.model_path,
            map_location="cpu",
            weights_only=True
        )

        if checkpoint.get("smoke_test", False):
            raise ValueError("임시 데이터로 학습된 모델입니다.")

        if checkpoint["target_labels"] != FINDING_NAMES:
            raise ValueError("모델의 소견 순서가 다릅니다.")

        if checkpoint["class_names"] != [
            "negative", "positive", "uncertain"
        ]:
            raise ValueError("모델의 클래스 순서가 다릅니다.")

        input_spec = checkpoint["input"]

        if (
            input_spec["channels"] != 3
            or input_spec["height"] != input_spec["width"]
        ):
            raise ValueError(
                "3채널 정사각형 입력 모델이어야 합니다."
            )

        if not (
            np.allclose(input_spec["mean"], MEAN)
            and np.allclose(input_spec["std"], STD)
        ):
            raise ValueError(
                "학습 모델과 전처리의 정규화 설정이 다릅니다."
            )

        self.image_size = int(input_spec["height"])
        self.checkpoint_epoch = checkpoint["epoch"]

        self.model = CXRModel()
        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image, include_gradcam=False):
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                tensor = preprocess_image(
                    opened, image_size=self.image_size
                )
        elif isinstance(image, Image.Image):
            tensor = preprocess_image(
                image, image_size=self.image_size
            )
        else:
            raise TypeError(
                "이미지 파일 경로 또는 PIL 이미지를 입력하세요."
            )

        inputs = tensor.unsqueeze(0).to(self.device)
        self.model.eval()

        with torch.no_grad():
            logits = self.model(inputs)
            probabilities = (
                logits.softmax(dim=-1)[0].cpu().numpy()
            )

        predictions = probabilities.argmax(axis=-1)
        findings = []

        for i, finding in enumerate(FINDING_NAMES):
            findings.append({
                "finding": finding,
                "prediction": CLASS_NAMES[int(predictions[i])],
                "probabilities": {
                    "negative": float(probabilities[i, 0]),
                    "positive": float(probabilities[i, 1]),
                    "uncertain": float(probabilities[i, 2]),
                },
            })

        output = {"findings": findings}

        # 예측 클래스와 관계없이 양성 클래스 Grad-CAM 생성
        if include_gradcam:
            gradcam = GradCAM(self.model)

            try:
                output["gradcam"] = {
                    finding: gradcam.generate(
                        tensor,
                        finding_idx=i,
                        class_idx=1
                    )
                    for i, finding in enumerate(FINDING_NAMES)
                }
            finally:
                gradcam.close()
                self.model.zero_grad(set_to_none=True)

        return output


# ============================================================
# 전처리 이미지 및 Grad-CAM overlay 저장
# ============================================================

def save_gradcam_overlays(
    image_path, result, image_size, save_dir
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as original:
        tensor = preprocess_image(
            original, image_size=image_size
        )

    # 정규화 해제
    pixels = tensor.permute(1, 2, 0).cpu().numpy()
    pixels = np.clip(
        pixels * np.asarray(STD) + np.asarray(MEAN),
        0, 1
    )

    image_id = Path(image_path).stem

    processed_path = (
        save_dir / f"{image_id}_processed.png"
    )
    Image.fromarray(
        np.round(pixels * 255).astype(np.uint8)
    ).save(processed_path)

    saved_paths = {
        "processed_image": str(processed_path)
    }

    for finding, cam in result["gradcam"].items():
        heatmap = colormaps["jet"](cam)[..., :3]

        # 반응이 약한 부분은 원본이 더 잘 보이도록 표시
        alpha = (0.4 * cam)[..., None]
        overlay = np.clip(
            pixels * (1 - alpha) + heatmap * alpha,
            0, 1
        )

        filename = (
            f"{image_id}_{finding.replace(' ', '_')}.png"
        )
        path = save_dir / filename

        Image.fromarray(
            np.round(overlay * 255).astype(np.uint8)
        ).save(path)

        saved_paths[finding] = str(path)

    return saved_paths


# ============================================================
# 파일 직접 실행 시 테스트
# ============================================================

def main():
    import json
    import pandas as pd

    # 설정 및 테스트 이미지 확인
    if not TEST_CSV.is_file():
        raise FileNotFoundError(
            f"test CSV가 없습니다: {TEST_CSV}"
        )

    if not IMAGE_ROOT.is_dir():
        raise FileNotFoundError(
            f"이미지 폴더가 없습니다: {IMAGE_ROOT}"
        )

    test_df = pd.read_csv(
        TEST_CSV,
        dtype={"dicom_id": "string"}
    )

    if not 0 <= IMAGE_INDEX < len(test_df):
        raise ValueError(
            f"IMAGE_INDEX는 0~{len(test_df) - 1} 범위여야 합니다."
        )

    image_id = test_df.iloc[IMAGE_INDEX]["dicom_id"]
    image_path = IMAGE_ROOT / f"{image_id}.jpg"

    if not image_path.is_file():
        raise FileNotFoundError(
            f"이미지가 없습니다: {image_path}"
        )

    # 모델 불러오기
    predictor = CXRInference(model_path=MODEL_PATH)

    print("모델 불러오기 완료")
    print("사용 모델:", predictor.model_path)
    print("사용 장치:", predictor.device)
    print("입력 이미지:", image_path)

    # 예측 및 Grad-CAM 생성
    result = predictor.predict(
        image_path,
        include_gradcam=True
    )

    prediction_output = {"findings": result["findings"]}

    print(json.dumps(
        prediction_output,
        ensure_ascii=False,
        indent=2
    ))

    for finding, cam in result["gradcam"].items():
        print(
            f"{finding} Grad-CAM: "
            f"shape={cam.shape}, "
            f"range=({cam.min():.4f}, {cam.max():.4f})"
        )

    # 모델 실행 폴더명별 결과 저장
    save_dir = OUTPUT_ROOT / predictor.model_path.parent.name

    saved_paths = save_gradcam_overlays(
        image_path=image_path,
        result=result,
        image_size=predictor.image_size,
        save_dir=save_dir
    )

    # 예측 결과와 실행 정보 저장
    prediction_output["metadata"] = {
        "model_path": str(predictor.model_path.resolve()),
        "checkpoint_epoch": predictor.checkpoint_epoch,
        "image_path": str(image_path.resolve()),
        "test_csv": str(TEST_CSV.resolve()),
        "image_index": IMAGE_INDEX,
        "gradcam_target_class": "positive",
    }

    prediction_path = (
        save_dir / f"{image_id}_predictions.json"
    )
    prediction_path.write_text(
        json.dumps(
            prediction_output,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("이미지 저장 완료:")
    print(json.dumps(
        saved_paths,
        ensure_ascii=False,
        indent=2
    ))
    print("예측 결과 저장:", prediction_path)


if __name__ == "__main__":
    main()