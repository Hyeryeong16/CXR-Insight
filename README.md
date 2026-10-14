# PyCharm용 CXR 전처리 · Dataset · DataLoader

샘플 이미지 전처리, 크기 조정, 정규화, Dataset·DataLoader 준비를 구현했습니다.
기준은 `CXR_Preprocessing_Progress (1).md`와 3채널 전환 결정입니다.

## 반영한 기준

| 항목 | 구현 |
| --- | --- |
| 입력 목록 | 팀원이 선정한 `train.csv`, `val.csv`, `test.csv` |
| 촬영 방향 | AP·PA만 허용; 다른 방향이 있으면 CSV 확인 요청 |
| 이미지 처리 | 흑백 → 비율 유지 검은 패딩 → bilinear 224×224 → 동일 흑백값 3채널 복제 |
| 정규화 | float32 / 255 → ImageNet mean·std 적용 |
| 소견 순서 | Cardiomegaly → Edema → Pleural Effusion |
| U-multiclass | CSV 0→0, 1→1, -1→2; 정답 `torch.long` |
| 이미지 반환 | `[3, 224, 224]`, float32 |
| 배치 반환 | 이미지 `[B, 3, 224, 224]`, 정답 `[B, 3]` |
| split | 공식 split 유지; train만 shuffle=True |

원본 JPG/CSV는 수정하거나 저장하지 않고 읽을 때 처리합니다. `linked_df`, Drive 마운트, Colab 셀 실행 순서가 필요하지 않습니다.
다운로드와 전체 이미지 손상 검사는 팀원의 작업 범위로 두고, 이 코드는 필요한 파일의 존재와 실제 배치 읽기를 확인합니다.
미다운로드 파일은 전체 실행에서 조용히 제외하지 않고 오류로 알려줍니다.

## 1. 파일 배치

압축을 풀고 `cxr_preprocessing.py`, `requirements.txt`, 이 문서를 PyCharm 프로젝트 최상위에 놓으세요.
팀원이 제공한 실제 최종 선정 CSV는 아래 위치에 놓습니다.

| 파일 | 프로젝트 내 위치 |
| --- | --- |
| Python 코드 | `cxr_preprocessing.py` |
| Train 목록 | `data/processed/train.csv` |
| Validation 목록 | `data/processed/val.csv` |
| Test 목록 | `data/processed/test.csv` |
| 이미지 기본 폴더 | `data/images/` |

CSV 필수 컬럼:

```text
subject_id, study_id, dicom_id, ViewPosition, split,
Cardiomegaly, Edema, Pleural Effusion
```

`val.csv`의 split 값은 팀원 노트북의 공식 명칭인 `validate`여야 합니다.
CSV의 식별자는 문자열로 읽고 라벨은 숫자로 검증합니다. 최종 CSV에 남아 있는 NaN을 임의로 다시 0으로 바꾸지 않습니다.

## 2. PyCharm 환경 준비

프로젝트의 Python 인터프리터를 선택하세요. Python 3.11 또는 3.12 환경을 예로 사용할 수 있습니다.
PyCharm 하단 Terminal에서 실행합니다.

```bash
python -m pip install -r requirements.txt
```

이 전처리 코드는 torchvision 없이 실행됩니다. 나중에 DenseNet121 모델 학습을 할 때 torchvision을 추가하고,
GPU를 쓸 경우 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)에서 운영체제와 CUDA에 맞는 torch/torchvision 조합을 선택하세요.
PyCharm은 실행 환경이므로 PyCharm 사용 자체가 GPU 사용을 의미하지 않습니다.

## 3. 이미지 위치 설정 — 여기가 가장 중요

코드 맨 위의 `IMAGE_ROOT`는 실제 JPG를 보관한 폴더로 바꾸세요.

### MIMIC 원본 폴더 구조를 유지했다면

예시 파일 위치(설명용 가상 번호):

```text
D:/mimic-cxr-jpg/files/p10/p10000000/s50000000/example-dicom-id.jpg
```

설정:

```python
IMAGE_ROOT = Path(r"D:\mimic-cxr-jpg")  # files의 바로 위 폴더
IMAGE_LAYOUT = "mimic"
```

`IMAGE_ROOT`에 `files`까지 넣으면 경로에 `files/files`가 생깁니다.
다운로드 담당 팀원에게 원본 폴더 구조를 유지하는지 확인해 실제 구조에 맞추세요.

### 샘플 JPG를 한 폴더에 모았다면

```python
IMAGE_ROOT = Path(r"D:\cxr_samples")
IMAGE_LAYOUT = "flat"
```

이 경우 파일은 `D:/cxr_samples/{dicom_id}.jpg`에 있어야 합니다.
원본 파일명은 유지하세요. CSV의 `image_path` 컬럼이 이미 있어도 이 코드에서는 위 설정을 기준으로 새 경로를 생성합니다.

CSV가 외부 폴더에 있다면 별도로 지정할 수 있습니다.

```python
CSV_DIR = Path(r"C:\workspace\chest-xray-multilabel\data\processed")
```

## 4. 먼저 샘플 실행

```python
RUN_MODE = "sample"
SAMPLE_SPLIT = "train"
SAMPLE_SIZE = 4
BATCH_SIZE = 4
NUM_WORKERS = 0
SHOW_PREVIEW = True
```

`cxr_preprocessing.py`를 우클릭 → Run 하세요. 또는 Terminal에서:

```bash
python cxr_preprocessing.py
```

선택한 split의 CSV 한 개만 필요합니다. 해당 CSV에 포함된 이미지 중 내려받은 파일을 골라 최대 4장 확인합니다.
기존 3장 샘플이 최종 선정 CSV에 없다면 이 코드로 확인되지 않습니다. 최종 CSV에 있는 JPG를 준비하거나 샘플이 포함된 공식 split을 선택하세요.
불확실(-1) 라벨이 있는 로컬 이미지가 있으면 먼저 선택해 2로 바뀌는지 확인합니다.
없으면 없다고 알리며, 실제 불확실 영상 검증을 완료했다고 간주하지 않습니다.

이미지 창에서 왼쪽 원본과 오른쪽 전처리 결과를 비교하세요. 창을 닫으면 실행이 이어집니다.
화면에는 정규화를 되돌린 영상을 표시하며, 이미지나 환자별 CSV를 새 파일로 저장하지 않습니다.

예상 배치 형식(실제 배치 수는 준비된 이미지 수에 따라 달라집니다):

```text
이미지 배치 크기: torch.Size([3, 3, 224, 224])
이미지 자료형: torch.float32
라벨 배치 크기: torch.Size([3, 3])
라벨 자료형: torch.int64
```

ImageNet 정규화 이후 픽셀값은 음수가 되거나 1을 넘을 수 있습니다. `0~1`은 `/255` 직후의 범위입니다.
흑백값은 정규화 전 3채널에서 동일하지만, 채널별 mean/std가 달라 정규화 후에는 서로 다릅니다.
클래스 2는 판독문의 불확실 소견을 뜻합니다. 모델의 낮은 신뢰도와는 다른 의미입니다.
`[0,0,0]`도 세 목표 소견이 음성이라는 의미이며 모든 질환이 없는 정상 판정은 아닙니다.

## 5. 전체 이미지 다운로드 후 실행

```python
RUN_MODE = "full"
```

세 CSV에 있는 파일을 모두 연결하고 split 간 환자·검사·이미지 중복을 검사합니다.
선정 CSV가 보고서와 같다면 Train 10,000장, Validate 1,077장, Test 2,284장이 출력되어야 합니다.
장수는 하드코딩하지 않으며 CSV 내용이 기준입니다.
세 소견 모두 -1인 이미지 수도 split별로 출력합니다. Train에서 4장이라는 기록은 실제 CSV 출력으로 확인하세요.
파일 존재 확인만으로 모든 이미지의 손상 여부가 검증되지는 않습니다. 손상된 파일은 실제로 읽힐 때 경로와 함께 오류가 나므로 팀원의 전체 점검 결과도 확인하세요.

## 6. 모델 학습 코드와 연결

별도 `train.py`에서 아래처럼 사용하세요. Windows에서 `num_workers>0`으로 바꿀 때도 실행 코드는 main 가드 안에 둡니다.

```python
from multiprocessing import freeze_support
from cxr_preprocessing import (
    build_dataloaders, CSV_DIR, IMAGE_ROOT, IMAGE_LAYOUT
)


def main():
    loaders = build_dataloaders(
        csv_dir=CSV_DIR,
        image_root=IMAGE_ROOT,
        image_layout=IMAGE_LAYOUT,
        image_size=224,
        batch_size=4,
        num_workers=0,
    )
    train_loader = loaders["train"]
    val_loader = loaders["validate"]
    test_loader = loaders["test"]

    for images, labels in train_loader:
        print(images.shape, labels.shape)
        break  # 실제 학습 코드로 교체


if __name__ == "__main__":
    freeze_support()
    main()
```

이 파일은 모델 생성·학습·성능 평가를 하지 않습니다. 모델 쪽에서는 9개 logits를 `[B,3,3]`으로 바꾸고
소견별 `CrossEntropyLoss(logits[:, i, :], labels[:, i])`를 평균하는 방식으로 연결하세요.
손실 계산 전 sigmoid나 softmax를 적용하지 않습니다. 정규화는 이미 Dataset에서 끝났으므로 다시 적용하지 않습니다.

## 설계 이유와 근거

- 마크다운의 기존 1채널 `/255` 구현에서, 이전 대화의 DenseNet121용 3채널/ImageNet 정규화 결정으로 확장했습니다.
- [Torchvision DenseNet121 공식 문서](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.densenet121.html)의 mean/std를 사용합니다.
  다만 공식 기본 변환의 `256 resize → 224 center crop`과 달리, 이 프로젝트에서는 기존 합의대로 `검은 패딩 → 224 resize`를 사용해 전체 영상 범위를 유지합니다.
- [PyTorch Dataset·DataLoader 공식 문서](https://docs.pytorch.org/docs/stable/data.html)의 Dataset/DataLoader 인터페이스를 사용합니다.
- 학습/검증/테스트 모두 동일한 전처리입니다. 이번 담당 범위에는 랜덤 증강을 넣지 않았습니다.
- 코드 공유 시 실제 JPG·라벨 CSV·환자 식별자 출력·인증 정보를 저장소에 올리지 마세요. 데이터 파일은 각자의 접근권한으로 준비합니다.

## 검증 기록

실제 MIMIC JPG와 선정 CSV는 이 제작 환경에 제공되지 않았습니다.
동봉 코드의 검증 결과와 범위는 `VALIDATION.md`를 참고하세요.
