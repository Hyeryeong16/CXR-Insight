# DenseNet121 전체 데이터 학습 실험

심비대(Cardiomegaly), 폐부종(Edema), 흉수(Pleural Effusion)를 예측하는 U-ignore 실험 코드 모음입니다.
기존 ResNet50 파일럿 실험 이후의 전체 데이터 학습 코드를 정리했습니다.

## 포함된 코드

| 순서 | 파일 | 실험 목적 |
|---|---|---|
| 공통 | `cxr_preprocessing_uignore.py` | 기존 전처리 결과의 불확실 라벨 2를 -100으로 바꿔 해당 소견의 손실·지표 계산에서 제외 |
| 1 | `train_densenet121_uignore_full.py` | 가중치 없는 기본 U-ignore 학습 |
| 2 | `train_densenet121_uignore_weighted_full.py` | 학습 데이터의 클래스 빈도를 반영한 가중치 적용 |
| 3 | `train_densenet121_uignore_weighted_scheduler_full.py` | 가중치 학습에 학습률 감소 Scheduler 추가 및 검증 AUROC 보고 |

이 파일들은 보관된 원본과 바이트 단위로 동일합니다. GitHub에 정리하기 위해 학습 로직이나 경로를 수정하지 않았습니다. 실제 GPU 서버에서 추가로 수정한 버전이 있다면 업로드 전에 서버 파일과 비교하세요.

## 배치 위치와 의존성

이 README는 저장소의 `full_training/README.md`에 두고, Python 파일 4개는 **저장소 최상위**에 둡니다. 각 스크립트가 자신의 위치를 기준으로 CSV와 출력 경로를 계산하기 때문입니다.

저장소 최상위에는 기존 프로젝트의 `cxr_preprocessing.py`도 있어야 합니다. 이 파일은 이번 묶음에 포함되지 않았습니다. `cxr_preprocessing_uignore.py`가 이 파일의 `build_dataloaders`를 재사용합니다.

필요한 기존 환경은 Python, PyTorch, torchvision 및 기존 `cxr_preprocessing.py`가 사용하는 패키지입니다. ImageNet 사전학습 DenseNet121을 사용합니다.

기본 경로는 다음과 같습니다.

| 항목 | 코드의 기본값 |
|---|---|
| CSV | 저장소 최상위의 `data/processed/` |
| 이미지 | `/home/team30103/physionet_data` |
| 이미지 배치 방식 | `image_layout="flat"` |
| 기본 실험 출력 | `runs/uignore_full/train_.../` |
| 가중치 실험 출력 | `runs/uignore_weighted_full/train_.../` |
| 가중치+Scheduler 출력 | `runs/uignore_weighted_scheduler_full/train_.../` |

서버에서 실제로 사용한 이미지 폴더가 MIMIC의 환자/검사 하위 폴더 구조라면 `image_layout="mimic"`이 필요합니다. 현재 보관본은 세 학습 파일 모두 `flat`입니다. `CSV_DIR`, `IMAGE_ROOT`, 이미지 배치 방식은 실제 학습 환경과 일치시켜야 합니다. CSV 파일명과 열 규칙은 기존 `cxr_preprocessing.py`의 규칙을 따릅니다.

코드가 기대하는 DataLoader 키는 `train`, `validate`, `test`입니다. 기본 코드 설명의 데이터 수는 학습 9,999장, 검증 1,077장, 테스트 2,284장이며, 가중치 및 Scheduler 코드는 이 수를 명시적으로 검사합니다.

## 공통 설정

| 항목 | 보관본의 설정 |
|---|---|
| 모델 | ImageNet 사전학습 DenseNet121 |
| 입력 | RGB 3채널, 224 × 224 |
| 출력 | 소견 3개 × 음성·양성 2클래스 |
| 라벨 | 0=음성, 1=양성, -100=계산 제외 |
| Optimizer | AdamW |
| 초기 학습률 | 0.0001 |
| Weight decay | 0.0001 |
| Batch size | 4 |
| 최대 Epoch | 10 |
| Early stopping | 검증 손실이 연속 3회 개선되지 않으면 종료 |
| Seed | 42 |
| 최적 모델 선택 | 검증 손실 최소 |

불확실 라벨이 있는 이미지를 통째로 삭제하지 않고, 불확실한 소견만 계산에서 제외합니다. 모든 라벨이 무시 대상인 배치는 학습 업데이트를 건너뜁니다.

현재 코드는 모두 early stopping이 켜진 최대 10 epoch 버전입니다. 50 epoch 실험이나 early stopping 해제 실험의 코드로 간주하면 안 됩니다.

## 실험별 차이

### 1. 기본 U-ignore

유효한 0/1 라벨에 대해 가중치 없는 Cross Entropy를 계산합니다. 검증 손실 최소 모델을 저장합니다.

### 2. U-ignore + 클래스 가중치

훈련 라벨만으로 각 소견의 음성·양성 빈도를 계산합니다. 빈도의 역제곱근을 이용하고 소견별 평균 가중치가 1이 되도록 정규화합니다. 훈련 손실은 가중치를 적용하지만 검증 손실은 가중치 없는 Cross Entropy를 사용합니다.

### 3. U-ignore + 가중치 + Scheduler

2번 실험에 `ReduceLROnPlateau`를 추가합니다. 검증 손실이 연속 2회 개선되지 않으면 다음 epoch부터 학습률을 절반으로 줄입니다. `scheduler_patience=1`, `factor=0.5`, `min_lr=0.000001`입니다. 학습률을 줄여도 early stopping 카운터는 초기화하지 않습니다.

소견별 양성 softmax 확률로 검증 AUROC를 계산합니다. 불확실 라벨은 소견별로 제외합니다. AUROC는 보고용이며 모델 선택 기준은 검증 손실 최소입니다.

## 실행

저장소 최상위에서, 기존 학습 가상환경을 활성화한 상태로 실행합니다. 업로드만 하려면 재학습은 필요하지 않습니다.

```bash
# 1. 기본 U-ignore
python train_densenet121_uignore_full.py

# 2. 클래스 가중치
python train_densenet121_uignore_weighted_full.py

# 3. 클래스 가중치 + Scheduler
python train_densenet121_uignore_weighted_scheduler_full.py
```

가중치 및 Scheduler 코드에 포함된 CPU 검사는 이미지 없이 실행할 수 있습니다. PyTorch는 필요합니다.

```bash
python train_densenet121_uignore_weighted_full.py --self-test
python train_densenet121_uignore_weighted_scheduler_full.py --self-test
```

Scheduler 코드로 종료된 실험을 추가 평가하려면 실제 실행 폴더를 지정합니다.

```bash
python train_densenet121_uignore_weighted_scheduler_full.py --evaluate-run runs/uignore_weighted_scheduler_full/실제_실행_폴더
```

가중치 코드의 `BASELINE_DIR`은 과거 기본 모델 실행 폴더를 가리킵니다. `--baseline` 또는 비교 출력이 필요하면 자신의 실행 폴더로 맞춥니다.

## 지표와 결과 정리

- Macro F1: 소견마다 음성·양성 F1을 평균한 뒤 세 소견을 평균합니다.
- 양성 평균 F1: 세 소견의 양성 클래스 F1을 평균합니다. 가중치 및 Scheduler 코드는 이 값을 직접 보고합니다.
- AUROC: Scheduler 코드에서 검증 세트 전체를 기준으로 소견별 계산합니다.
- U-multiclass의 3클래스 Macro F1과 U-ignore의 2클래스 Macro F1은 집계 대상이 다릅니다.
- 테스트 세트는 학습이나 체크포인트 선택에 사용하지 않습니다. 이 학습 스크립트들만으로 최종 테스트 평가가 완료되는 것은 아닙니다.

각 실행 폴더에는 `config.json`, `history.csv`, `best_val_metrics.json`, `best_model.pth`가 저장됩니다. 최종 비교표는 실제 실행 폴더의 기록으로 작성하세요. 코드에 들어 있는 과거 비교 수치는 저장된 참고값이며 새 실행 결과가 아닙니다.

현재 묶음에는 실제 실행 로그와 체크포인트가 없어 실험 간 우열이나 최종 모델을 확정하지 않았습니다.

## 검증 범위

정리 과정에서 Python 파일 4개의 문법과 원본 일치 여부를 확인했습니다. 정리 환경에는 PyTorch/GPU와 학습 데이터가 없어 학습 실행 및 내장 CPU 검사는 수행하지 않았습니다.
