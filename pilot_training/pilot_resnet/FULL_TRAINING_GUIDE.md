# Pilot 학습에서 전체 데이터 학습으로 전환하기

흉부 X-ray의 심비대(Cardiomegaly), 폐부종(Edema), 흉수(Pleural Effusion)를 분류하는 모델을 전체 데이터로 학습하기 위한 전환 가이드입니다.

> 이 문서는 기존 코드에서 수정할 부분을 정리한 안내입니다. 전체 데이터용 학습 코드가 이미 수정·실행되었다는 의미는 아닙니다. 경로와 파일명 예시는 실제 서버 환경에 맞춰 변경해야 합니다.

## 1. 전환 원칙

모델을 새로 구현할 필요는 없습니다. 선택한 모델 구조와 라벨 처리 방식은 유지하고, 데이터 연결 및 결과 저장 설정을 전체 데이터에 맞게 변경합니다.

**전체 데이터 학습은 전체 데이터 중 train에 배정된 이미지를 모두 학습한다는 뜻입니다.** Validation과 test 이미지를 train에 합치지 않습니다.

| 구분 | 역할 |
|---|---|
| Train | 손실을 계산하고 모델 가중치를 업데이트 |
| Validation | best 모델 선택, 조기 종료, 모델·설정 비교 |
| Test | 모델과 설정 선택을 마친 뒤 최종 성능 평가 |

## 2. 변경할 항목

| 항목 | 기존 pilot 설정 | 전체 데이터 학습에서 필요한 변경 |
|---|---|---|
| CSV 폴더 | `data/processed/pilot` | 전체 데이터용 분할 CSV가 있는 폴더로 변경 |
| CSV 파일명 | `pilot_train.csv`, `pilot_val.csv`, `pilot_test.csv` | 실제 전체 데이터 CSV 파일명과 연결 |
| 이미지 경로 | pilot 이미지 저장 위치 | 전체 이미지가 저장된 위치로 연결 |
| 이미지 폴더 구조 | `image_layout="flat"` | 실제 이미지 경로를 해석할 수 있도록 전처리 모듈 설정 확인 |
| 결과 저장 위치 | `runs/pilot` 등 | 전체 학습 전용 폴더로 분리 |
| 이미지 수 검사 | 일부 ResNet 코드에서 합계 1,000장 확인 | pilot 전용 제한을 제거하고 전체 split 검증으로 대체 |
| 배치 크기·로딩 설정 | `batch_size=4`, `num_workers=0` | 우선 유지 가능. GPU 메모리·서버 자원에 맞춰 조정 |

### CSV 설정 예시

다음은 설정 변경 예시이며, 완성된 학습 스크립트가 아닙니다.

```python
from pathlib import Path

PROJECT_DIR = Path("/path/to/chest-xray-multilabel")
CSV_DIR = PROJECT_DIR / "data" / "processed" / "full"

TRAIN_CSV = CSV_DIR / "train.csv"
VAL_CSV = CSV_DIR / "val.csv"
TEST_CSV = CSV_DIR / "test.csv"

IMAGE_ROOT = Path("/path/to/full_images")
OUTPUT_ROOT = PROJECT_DIR / "runs" / "full_densenet121_umulticlass"
```

**상단의 경로 변수만 바꾸면 충분한지는 DataLoader 구현에 따라 다릅니다.**

- 기존 DenseNet `train_pilot.py`는 `build_dataloaders(csv_dir=CSV_DIR, ...)`로 폴더를 전달합니다. 실제로 읽는 파일명은 `cxr_preprocessing.py`에서 확인해야 합니다.
- ResNet pilot 코드는 CSV 파일명 별칭을 연결하는 로직과 합계 1,000장 검사를 포함합니다. 전체 데이터 연결 시 이 부분도 함께 수정해야 합니다.
- CSV 열 이름, 라벨 값, 이미지 경로 표현이 기존 Dataset이 기대하는 형식과 일치해야 합니다.

### 이미지 경로 확인

현재 DenseNet pilot 코드는 `image_layout="flat"`을 사용합니다. 전체 이미지를 환자·검사별 하위 폴더에 저장했다면 이 설정이 맞는지 확인해야 합니다.

전처리 모듈이 지원하는 설정과 CSV의 경로 열을 확인한 뒤 연결합니다. 지원 여부를 확인하지 않고 다른 `image_layout` 값을 임의로 넣지 않습니다.

### Pilot 전용 이미지 수 제한

ResNet pilot 코드에는 다음과 같은 검사가 있습니다.

```python
if total != 1000:
    raise ValueError(...)
```

전체 학습에서는 이 조건을 제거하거나 전체 데이터에 맞게 변경합니다. 대신 split별 행 수, 이미지 누락, 중복, 환자 중복 여부를 확인합니다.

## 3. 유지할 수 있는 항목

| 항목 | 전환 시 원칙 |
|---|---|
| 모델 구조 | 선택한 DenseNet121 또는 ResNet50 유지 |
| 전처리 | 기존 크기 조정·패딩·3채널 변환·정규화 유지 |
| 라벨 처리 | 선택한 U-multiclass 또는 U-ignore 유지 |
| 최적화 설정 | 기존 학습률·weight decay로 시작 가능 |
| best 모델 선택 | 기존 최저 validation loss 기준 유지 가능 |
| 조기 종료 | 기존 `patience=3` 유지 가능 |
| 최대 epoch | 기존 10 epoch로 시작 가능. 데이터가 늘었다고 반드시 늘릴 필요는 없음 |

전체 데이터에서는 한 epoch의 배치 수와 학습 시간이 증가합니다. Pilot에서 2 epoch가 best였더라도 전체 학습의 best epoch가 같다고 가정하지 않습니다.

배치 크기를 바꾸면 학습 조건도 달라집니다. 비교 실험에서는 가능한 한 같은 설정을 사용하고, 변경 사항은 기록합니다.

## 4. 라벨 처리 방식 확인

| 방식 | 학습 대상 | 평가·연결 시 주의점 |
|---|---|---|
| U-multiclass | 음성·양성·불확실 | 현재 코드에서 원본 `-1`을 불확실 클래스 `2`로 변환. 소견별 3클래스 출력 |
| U-ignore | 음성·양성 | 불확실한 소견만 손실에서 제외. 같은 이미지의 다른 확정 소견은 사용 |

U-ignore의 출력 구현은 코드마다 다를 수 있습니다. 현재 ResNet50 U-ignore 코드는 `(B, 3, 2)`를 출력하며, 양성 확률은 `softmax(dim=-1)[..., 1]`로 계산합니다. 다른 팀원의 모델은 소견당 logit 하나와 sigmoid를 사용할 수 있으므로 평가 코드가 해당 체크포인트 구조와 맞아야 합니다.

서로 다른 라벨 처리 방식의 loss와 원래 Macro F1을 그대로 비교하지 않습니다. 네 모델의 공통 비교에서는 소견별로 확정 정답만 포함하고, 양성 확률 계산 및 판정 규칙을 명시해 같은 평가 기준을 적용합니다.

## 5. 새 학습과 이어서 학습 구분

현재 확인한 DenseNet pilot 코드는 `CXRModel(pretrained=True)`로 모델을 생성합니다.

- **현재 방식 유지:** ImageNet 사전학습 가중치에서 전체 데이터 학습을 새로 시작합니다.
- **Pilot에서 이어서 학습:** pilot의 `best_model.pth`를 명시적으로 불러오는 별도 수정이 필요합니다.

`pretrained=True`만으로 pilot 체크포인트가 자동으로 불러와지는 것은 아닙니다. 비교 실험에서는 초기화 방식도 통일하고 기록합니다.

## 6. 실행 전 확인 목록

- [ ] 전체 데이터용 train·val·test CSV가 준비되어 있다.
- [ ] 같은 환자의 이미지가 서로 다른 split에 겹치지 않는다.
- [ ] Pilot과 전체 데이터의 분할 관계를 확인했다. 이전 학습에 사용한 환자가 새 test에 섞이지 않도록 관리한다.
- [ ] CSV의 이미지가 실제로 존재하고 경로가 올바르게 해석된다.
- [ ] CSV 열 이름·라벨 인코딩·결측값 처리가 Dataset과 일치한다.
- [ ] split별 이미지 수와 소견별 클래스 분포를 확인했다.
- [ ] Pilot 전용 1,000장 제한을 변경했다.
- [ ] 결과 저장 위치를 pilot 결과와 분리했다.
- [ ] 실제 이미지 한 배치를 읽어 이미지·라벨·모델 출력 형태를 확인했다.
- [ ] 전처리 설정, 데이터 분할, 학습 설정과 모델 초기화 방식을 기록한다.

소수 배치로 이미지 로딩과 GPU 연산을 먼저 확인한 뒤 전체 학습을 시작합니다. GPU 메모리가 부족하면 배치 크기를 조정합니다.

## 7. 학습 후 평가 담당자에게 전달할 정보

- 저장된 `best_model.pth`
- 모델 구조를 정의한 코드와 필요한 전처리 코드
- 모델 종류와 라벨 처리 방식
- 소견 순서와 클래스 인코딩
- 이미지 전처리·정규화 설정
- `config.json`, `history.csv`, `best_val_metrics.json` 등 해당 실행의 설정·결과
- 사용한 데이터 분할과 체크포인트 경로

모델 선택에는 validation을 사용합니다. 선택을 마친 뒤 남겨둔 test로 최종 평가하며, test 결과를 보고 반복적으로 설정을 조정하지 않습니다.
