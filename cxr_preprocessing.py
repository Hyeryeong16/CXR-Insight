"""MIMIC-CXR-JPG 전처리와 U-multiclass Dataset / DataLoader 준비.

PyCharm에서 이 파일을 Run 하세요. 먼저 아래 '사용자 설정'을 수정하세요.
모델 학습 파일에서는 build_dataloaders()를 import 해서 사용할 수 있습니다.
원본 JPG와 CSV는 변경하지 않습니다.
"""

from pathlib import Path
from multiprocessing import freeze_support

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


# 1. 사용자 설정 ---------------------------------------------------------------
# 이 파일을 프로젝트 최상위에 놓으면 기본 경로를 그대로 사용할 수 있습니다.
# 외부 폴더를 쓰려면 예: IMAGE_ROOT = Path(r"D:\mimic-cxr-jpg")
PROJECT_DIR = Path(__file__).resolve().parent
CSV_DIR = PROJECT_DIR / "data" / "processed"
IMAGE_ROOT = PROJECT_DIR / "data" / "images"

# mimic: IMAGE_ROOT / files / pXX / p환자번호 / s검사번호 / dicom_id.jpg
# flat:  IMAGE_ROOT / dicom_id.jpg (샘플을 한 폴더에 모아둔 경우)
IMAGE_LAYOUT = "mimic"

RUN_MODE = "sample"          # sample: 내려받은 일부 이미지 확인 / full: 전체 연결
SAMPLE_SPLIT = "train"       # train / validate / test
SAMPLE_SIZE = 4
IMAGE_SIZE = 224
BATCH_SIZE = 4
NUM_WORKERS = 0              # PyCharm / Windows에서는 먼저 0으로 확인하세요.
SHOW_PREVIEW = True          # 이미지 창을 닫으면 실행이 이어집니다.
SEED = 42

TARGET_LABELS = ["Cardiomegaly", "Edema", "Pleural Effusion"]
CLASS_MAP = {0: 0, 1: 1, -1: 2}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CSV_NAMES = {"train": "train.csv", "validate": "val.csv", "test": "test.csv"}


# 2. 최종 CSV 읽기와 실제 파일 경로 연결 --------------------------------------
def validate_labels(df):
    """최종 CSV에서 NaN을 다시 0으로 바꾸지 않고 잘못된 값을 알립니다."""
    missing = set(TARGET_LABELS) - set(df.columns)
    if missing:
        raise ValueError(f"라벨 컬럼이 없습니다: {sorted(missing)}")
    labels = df[TARGET_LABELS].apply(pd.to_numeric, errors="raise")
    if labels.isna().any().any():
        raise ValueError("최종 CSV에 NaN 라벨이 있습니다. 팀원 최종 라벨표를 확인하세요.")
    if not labels.isin(list(CLASS_MAP)).all().all():
        raise ValueError("원본 라벨은 0, 1, -1만 가능합니다. 이미 2로 변환된 CSV도 확인하세요.")
    return labels.astype("int64")


def read_split_csv(csv_dir, image_root, split, image_layout="mimic"):
    """이미 라벨이 연결된 최종 선정 CSV를 읽어 image_path를 추가합니다."""
    if split not in CSV_NAMES:
        raise ValueError(f"split은 {list(CSV_NAMES)} 중 하나여야 합니다.")
    if image_layout not in {"mimic", "flat"}:
        raise ValueError("IMAGE_LAYOUT은 'mimic' 또는 'flat'이어야 합니다.")
    csv_path = Path(csv_dir) / CSV_NAMES[split]
    if not csv_path.is_file():
        raise FileNotFoundError(f"최종 선정 CSV가 없습니다: {csv_path}")

    # 식별자를 문자열로 읽어 환자/검사 번호에 .0이 붙는 일을 방지합니다.
    df = pd.read_csv(csv_path, dtype={
        "subject_id": "string", "study_id": "string", "dicom_id": "string"
    })
    required = {"subject_id", "study_id", "dicom_id", "ViewPosition", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name}의 필수 컬럼이 없습니다: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"{csv_path.name}에 이미지 목록이 없습니다.")

    for column in ["subject_id", "study_id", "dicom_id"]:
        df[column] = df[column].str.strip()
        if df[column].isna().any() or df[column].eq("").any():
            raise ValueError(f"{csv_path.name}: {column}에 빈 식별자가 있습니다.")
    for column in ["subject_id", "study_id"]:
        if not df[column].str.fullmatch(r"[0-9]+").all():
            raise ValueError(f"{column}는 정수 번호여야 합니다. CSV에 .0 등이 있는지 확인하세요.")
    if not df["dicom_id"].str.fullmatch(r"[A-Za-z0-9-]+").all():
        raise ValueError("dicom_id에는 확장자나 경로 대신 원본 이미지 식별자를 넣으세요.")
    if df["dicom_id"].duplicated().any():
        raise ValueError(f"{csv_path.name}: dicom_id가 중복되어 있습니다.")
    if not df["ViewPosition"].isin(["AP", "PA"]).all():
        raise ValueError(f"{csv_path.name}: AP/PA 이외의 영상이 있습니다. 선정 CSV를 확인하세요.")
    if not df["split"].eq(split).all():
        raise ValueError(f"{csv_path.name}: split 값이 '{split}'과 일치하지 않습니다.")
    df[TARGET_LABELS] = validate_labels(df)

    image_root = Path(image_root).expanduser().resolve()
    paths = []
    for row in df.itertuples(index=False):
        if image_layout == "flat":
            path = image_root / f"{row.dicom_id}.jpg"
        else:
            path = (image_root / "files" / f"p{row.subject_id[:2]}"
                    / f"p{row.subject_id}" / f"s{row.study_id}" / f"{row.dicom_id}.jpg")
        paths.append(str(path))
    # CSV의 기존 경로 대신 이 컴퓨터의 IMAGE_ROOT를 기준으로 생성합니다.
    df["image_path"] = paths
    return df.reset_index(drop=True)


# 3. 흑백 변환 → 검은 패딩 → 224 크기 → 3채널 → 정규화 -------------------------
def resize_with_padding(image, image_size=224):
    """긴 변 기준 정사각형에 가운데 배치한 뒤 축소해 비율을 유지합니다."""
    if image_size <= 0:
        raise ValueError("image_size는 양수여야 합니다.")
    gray = image.convert("L")
    width, height = gray.size
    side = max(width, height)
    canvas = Image.new("L", (side, side), color=0)
    canvas.paste(gray, ((side - width) // 2, (side - height) // 2))
    return canvas.resize((image_size, image_size), Image.Resampling.BILINEAR)


def preprocess_image(image, image_size=224):
    resized = resize_with_padding(image, image_size)
    pixels = np.array(resized, dtype=np.float32) / 255.0
    # [H, W] → [1, H, W] → [3, H, W]; 색을 추가하는 것이 아닌 동일 흑백값 복제.
    tensor = torch.from_numpy(pixels).unsqueeze(0).repeat(3, 1, 1)
    mean = tensor.new_tensor(MEAN).view(3, 1, 1)
    std = tensor.new_tensor(STD).view(3, 1, 1)
    return (tensor - mean) / std


def denormalize_image(tensor):
    """화면 표시용: ImageNet 정규화를 되돌려 0~1 영상으로 만듭니다."""
    mean = tensor.new_tensor(MEAN).view(3, 1, 1)
    std = tensor.new_tensor(STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


# 4. Dataset: 이미지 한 장과 세 소견의 클래스 번호 반환 -------------------------
class CXRDataset(Dataset):
    def __init__(self, dataframe, image_size=224):
        if dataframe.empty:
            raise ValueError("Dataset에 넣을 이미지가 없습니다.")
        if "image_path" not in dataframe.columns:
            raise ValueError("image_path가 없습니다. read_split_csv()를 먼저 실행하세요.")
        self.df = dataframe.reset_index(drop=True).copy()
        self.image_size = image_size
        raw = validate_labels(self.df)
        mapped = raw.replace(CLASS_MAP).to_numpy(dtype=np.int64)
        self.labels = torch.from_numpy(mapped)  # [N, 3], torch.long

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_path = Path(self.df.iloc[index]["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"이미지 파일이 없습니다: {image_path}")
        try:
            with Image.open(image_path) as image:
                image_tensor = preprocess_image(image, self.image_size)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"이미지 읽기/전처리에 실패했습니다: {image_path}") from exc
        return image_tensor, self.labels[index]


# 5. 전체 split별 DataLoader (전체 다운로드 완료 후 사용) -----------------------
def build_dataloaders(csv_dir, image_root, image_layout="mimic", image_size=224,
                      batch_size=4, num_workers=0, seed=42):
    frames = {
        split: read_split_csv(csv_dir, image_root, split, image_layout)
        for split in CSV_NAMES
    }
    # 공식 split을 다시 나누지 않습니다. 환자/검사/이미지 중복이 있으면 중단합니다.
    pairs = [("train", "validate"), ("train", "test"), ("validate", "test")]
    for first, second in pairs:
        for column in ["subject_id", "study_id", "dicom_id"]:
            overlap = set(frames[first][column]) & set(frames[second][column])
            if overlap:
                raise ValueError(f"{first}–{second}: {column} 중복 {len(overlap)}개")

    loaders = {}
    for split, df in frames.items():
        exists = df["image_path"].map(lambda p: Path(p).is_file())
        if not exists.all():
            example = df.loc[~exists, "image_path"].iloc[0]
            raise FileNotFoundError(
                f"{split}: 미다운로드 이미지 {(~exists).sum()}장. 예: {example}\n"
                "전체 연결은 다운로드 완료 후 실행하세요. 샘플 확인은 RUN_MODE='sample'로 하세요."
            )
        dataset = CXRDataset(df, image_size=image_size)
        loaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
            drop_last=False, generator=torch.Generator().manual_seed(seed)
        )
    return loaders


# 6. 샘플 확인: 최종 CSV에 있고 실제로 내려받은 이미지 중 일부만 사용 -----------
def build_sample_dataset(csv_dir, image_root, split="train", image_layout="mimic",
                         sample_size=4, image_size=224):
    if sample_size < 1:
        raise ValueError("SAMPLE_SIZE는 1 이상이어야 합니다.")
    df = read_split_csv(csv_dir, image_root, split, image_layout)
    exists = df["image_path"].map(lambda p: Path(p).is_file())
    available = df.loc[exists].copy()
    print(f"{split} 선정 {len(df):,}장 / 로컬에 존재 {len(available):,}장")
    if available.empty:
        raise FileNotFoundError(
            "최종 CSV와 연결되는 로컬 JPG가 없습니다. IMAGE_ROOT / IMAGE_LAYOUT을 확인하세요.\n"
            f"첫 예상 경로: {df['image_path'].iloc[0]}"
        )

    # 가능한 경우 실제 -1 라벨 샘플을 앞에 넣어 클래스 2 변환도 확인합니다.
    uncertain = available[TARGET_LABELS].eq(-1).any(axis=1)
    ordered = pd.concat([available.loc[uncertain], available.loc[~uncertain]])
    sample = ordered.head(sample_size).reset_index(drop=True)
    if not uncertain.any():
        print("현재 로컬 이미지에는 원본 -1 라벨이 없습니다. 실제 불확실 샘플 검증은 추후 필요합니다.")
    print("주의: 이 샘플 Dataset은 동작 확인용이며 전체 학습/성능 평가용이 아닙니다.")
    return CXRDataset(sample, image_size=image_size)


def inspect_batch(dataset, batch_size=4, show_preview=True):
    # 그림과 CSV 행 순서를 일치시키기 위해 확인용 loader는 항상 shuffle=False.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    images, labels = next(iter(loader))
    print("이미지 배치 크기:", images.shape)
    print("이미지 자료형:", images.dtype)
    print("정규화 후 이미지 범위:", float(images.min()), "~", float(images.max()))
    print("라벨 배치 크기:", labels.shape)
    print("라벨 자료형:", labels.dtype)
    print("변환된 정답:\n", labels)
    # 정상값/양성/불확실의 순서가 아닌 음성/양성/불확실의 순서입니다.
    raw = dataset.df[TARGET_LABELS]
    print("세 소견 모두 불확실한 이미지 수:", int(raw.eq(-1).all(axis=1).sum()))
    for i in range(len(images)):
        row = dataset.df.iloc[i]
        original = row[TARGET_LABELS].astype(int).tolist()
        print(f"{row['dicom_id']} | 원본 {original} → 클래스 {labels[i].tolist()}")

    if show_preview:
        figure, axes = plt.subplots(len(images), 2, figsize=(9, 4 * len(images)), squeeze=False)
        for i in range(len(images)):
            row = dataset.df.iloc[i]
            with Image.open(row["image_path"]) as image:
                axes[i, 0].imshow(image.convert("L"), cmap="gray", vmin=0, vmax=255)
            # 정규화된 값을 직접 그리지 않고 먼저 정규화를 되돌립니다.
            gray = denormalize_image(images[i]).mean(dim=0).numpy()
            axes[i, 1].imshow(gray, cmap="gray", vmin=0, vmax=1)
            axes[i, 0].set_title(f"Original | {row['ViewPosition']}")
            axes[i, 1].set_title(f"Processed | classes={labels[i].tolist()}")
            for axis in axes[i]:
                axis.axis("off")
        figure.suptitle("Label order: Cardiomegaly / Edema / Pleural Effusion")
        figure.tight_layout()
        plt.show()
        plt.close(figure)


# 7. PyCharm의 Run 버튼으로 실행되는 부분 --------------------------------------
def main():
    # 가상 값으로 변환 규칙 확인. 실제 환자 이미지 검증과는 구분합니다.
    assert [CLASS_MAP[value] for value in [1, -1, 0]] == [1, 2, 0]
    if RUN_MODE == "sample":
        dataset = build_sample_dataset(
            CSV_DIR, IMAGE_ROOT, SAMPLE_SPLIT, IMAGE_LAYOUT, SAMPLE_SIZE, IMAGE_SIZE
        )
        inspect_batch(dataset, BATCH_SIZE, SHOW_PREVIEW)
    elif RUN_MODE == "full":
        loaders = build_dataloaders(
            CSV_DIR, IMAGE_ROOT, IMAGE_LAYOUT, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, SEED
        )
        for split, loader in loaders.items():
            print(f"\n{split}: 전체 {len(loader.dataset):,}장")
            inspect_batch(loader.dataset, BATCH_SIZE, SHOW_PREVIEW and split == "train")
        print("\n전체 split별 DataLoader 준비 완료. build_dataloaders()를 학습 코드에서 사용하세요.")
    else:
        raise ValueError("RUN_MODE는 'sample' 또는 'full'이어야 합니다.")


if __name__ == "__main__":
    freeze_support()
    main()
