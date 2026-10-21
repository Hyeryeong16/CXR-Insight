# DenseNet121 모델 준비 및 동작 확인

`model_training.py`는 흉부 X-ray에서 심비대·폐부종·흉수를 예측하기 위한 DenseNet121 모델과 학습·검증·저장 함수를 정의한 초기 모델 준비 코드입니다.

이 문서는 첨부된 `model_training (1).py`의 내용을 기준으로 작성했습니다. GitHub에서는 파일명을 `model_training.py`로 정리해 사용할 수 있습니다.

## 목적

- 소견별 음성·양성·불확실을 예측하는 U-multiclass 모델 구현.
- 이미지와 라벨의 입력 규격 확인.
- 학습·검증 루프 및 체크포인트 저장·복원 확인.
- Dataset 담당자가 준비한 DataLoader와 연결할 공통 학습 함수 제공.

기본 실행은 가상 이미지와 임의 라벨로 수행하는 **동작 확인(smoke test)**입니다. 실제 환자 데이터에 대한 학습 결과나 성능평가를 제공하지 않습니다.

## 모델 및 라벨 구성

| 항목 | 내용 |
|---|---|
| Backbone | DenseNet121 |
| 실제 학습 모델 초기화 | `CXRModel(pretrained=True)`로 ImageNet 가중치 사용 |
| 기본 동작 확인 | `CXRModel(pretrained=False)`로 무작위 초기화 |
| 입력 크기 | 3채널 224 × 224 |
| 소견 순서 | Cardiomegaly → Edema → Pleural Effusion |
| 클래스 순서 | negative → positive → uncertain |
| 원본 라벨 매핑 | 0→0, 1→1, -1→2 |
| 모델 출력 | `[배치 크기, 소견 3개, 클래스 3개]` |
| 손실 함수 | 소견별 3클래스 Cross Entropy의 평균 |
| 예측 확률 | 소견별 softmax |

세 소견을 각각 분류하므로 하나의 이미지에 여러 소견이 함께 양성으로 예측될 수 있습니다. 불확실 라벨도 세 번째 클래스로 학습합니다. 9개 출력을 하나의 9클래스 문제로 처리하지 않습니다.

## 설치와 기본 실행

기존 PyTorch 환경이 있다면 해당 환경을 사용합니다. 새 환경에서는 호환되는 PyTorch와 torchvision이 필요하며, GPU 환경에서는 서버의 CUDA 환경에 맞춰 설치합니다.

파일이 있는 폴더에서 실행합니다.

```bash
# 첨부 파일명을 유지한 경우
python "model_training (1).py"

# model_training.py로 이름을 정리한 경우
python model_training.py
```

패키지가 설치된 뒤 기본 동작 확인에는 CSV, 실제 이미지, ImageNet 가중치 다운로드가 필요하지 않습니다. CUDA가 있으면 GPU를 사용하고, 없으면 CPU로 실행합니다.

## 기본 실행에서 확인하는 항목

`__main__`에서 `smoke_check()`를 실행합니다.

1. 가상 흑백 이미지 2장과 임의 라벨을 생성합니다. 학습·검증용 가상 DataLoader를 각각 만듭니다.
2. 흑백 이미지를 3채널로 복제하고 정규화합니다.
3. 배치의 크기·자료형·라벨 범위와 모델 출력 형태를 검사합니다.
4. 가상 데이터로 1 epoch 학습·검증 후 임시 체크포인트를 저장합니다.
5. 저장 모델을 복원하고 복원 전후 예측값이 같은지 확인합니다.
6. 각 소견의 음성·양성·불확실 확률 합이 1인지 확인합니다.

코드에서 출력하도록 정의된 형태는 다음과 같습니다. 실제 실행 결과를 옮긴 것은 아닙니다.

```text
이미지: (2, 3, 224, 224) torch.float32
정답: (2, 3) torch.int64
모델 출력: (2, 3, 3)
동작 확인 성공: 학습 1회, 검증, 저장, 복원, 확률 합 확인
```

## 실제 데이터 연결 규격

실제 CSV 읽기·이미지 파일 연결·전처리·데이터 분할은 이 파일에 구현되어 있지 않습니다. 별도의 Dataset·DataLoader가 다음 규격을 충족해야 합니다.

| 항목 | 요구 규격 |
|---|---|
| images | `[B, 3, 224, 224]`, `torch.float32`, 유한한 값 |
| labels | `[B, 3]`, `torch.long` |
| 라벨 값 | 0=음성, 1=양성, 2=불확실 |
| 소견 순서 | Cardiomegaly, Edema, Pleural Effusion |
| 정규화 평균 | `[0.485, 0.456, 0.406]` |
| 정규화 표준편차 | `[0.229, 0.224, 0.225]` |

`CLASS_MAP`은 라벨 매핑 규칙을 정의하지만 실제 원본 CSV의 -1을 자동 변환하지는 않습니다. Dataset에서 -1을 2로 변환해야 합니다. 전처리는 비율 유지·정사각 패딩·크기 조정 및 정규화 규격에 맞춰 준비합니다. `check_batch()`는 배치 형태와 값 범위를 검사하며 실제 이미지 처리 방식을 보증하는 함수는 아닙니다.

파일을 `model_training.py`로 저장하고 실제 DataLoader를 준비한 뒤, 별도 학습 스크립트에서 다음과 같이 연결합니다. 아래 예시의 `train_loader`, `val_loader`는 실제 데이터로 미리 생성해야 합니다. 재현성을 위해 DataLoader 생성 전에도 `set_seed(42)`를 호출합니다.

```python
from model_training import (
    CXRModel, TrainConfig, choose_device, set_seed,
    check_batch, fit, load_checkpoint, predict_probabilities,
)

# 앞선 단계에서 실제 train_loader, val_loader를 준비한 상태
set_seed(42)
device = choose_device()
check_batch(*next(iter(train_loader)))
check_batch(*next(iter(val_loader)))

model = CXRModel(pretrained=True).to(device)
config = TrainConfig(epochs=10, learning_rate=1e-4,
                     weight_decay=1e-4, patience=3)
_, best_path = fit(model, train_loader, val_loader, config, device)
best_model, checkpoint = load_checkpoint(best_path, device)

images, labels = next(iter(val_loader))
probabilities = predict_probabilities(best_model, images)
# probabilities.shape: [B, 3, 3]
```

ImageNet 가중치는 최초 사용 시 다운로드가 필요할 수 있습니다. `fit()`이 반환하는 모델은 마지막 epoch 상태이므로, 평가·예측에는 `load_checkpoint()`로 최적 체크포인트를 복원합니다.

## 학습 기본값과 평가

| 설정 | 기본값 |
|---|---|
| 최대 Epoch | 10 |
| 학습률 | 0.0001 |
| Weight decay | 0.0001 |
| Optimizer | AdamW |
| Early stopping | 검증 손실이 연속 3회 개선되지 않으면 종료 |
| 모델 선택 | 검증 손실 최소 |
| Seed | 42 |

기본 smoke test에서는 `TrainConfig(epochs=1)`을 사용하므로 1 epoch만 실행합니다. 실제 학습의 batch size는 전달받은 DataLoader가 결정합니다.

`run_epoch()`는 손실, Macro F1, 소견·클래스별 F1, 소견별 혼동행렬, 이미지 수를 반환합니다. Macro F1은 세 소견의 음성·양성·불확실 F1, 총 9개 값을 평균합니다. 관측·예측이 모두 없는 클래스의 F1은 0입니다.

가중치 손실·Scheduler·AUROC 계산은 포함하지 않습니다. `fit()`은 테스트 DataLoader를 받지 않으며 테스트 평가는 별도로 수행합니다.

## 저장 파일

기본 출력 폴더는 이 Python 파일과 같은 위치의 `runs/`입니다. `TrainConfig.output_root`로 변경할 수 있습니다.

| 파일 | 내용 |
|---|---|
| `config.json` | 모델·라벨·입력 규격 및 실행 설정 |
| `history.csv` | epoch별 학습·검증 손실과 Macro F1 |
| `best_val_metrics.json` | 검증 손실 최소 시점의 검증 지표 |
| `demo_checkpoint.pth` | smoke test용 임시 체크포인트 |
| `best_model.pth` | 실제 학습용 최적 체크포인트 |

가상 데이터 실행 결과는 `runs/smoke_날짜시간/`, 실제 학습 결과는 `runs/train_날짜시간/`에 생성됩니다. 임시 체크포인트에는 `smoke_test=True`가 기록되고, 기본 `load_checkpoint()`는 이를 차단합니다. 동작 확인에서만 `allow_demo=True`로 복원합니다.

## train_full.py와의 관계

| 항목 | `model_training.py` | `train_full.py` |
|---|---|---|
| 역할 | 초기 모델 정의·동작 확인·공통 함수 제공 | 서버의 실제 데이터 연결 및 학습 실행 |
| 직접 실행 | 가상 데이터 1 epoch | 실제 데이터로 최대 10 epoch |
| 실제 데이터 연결 | 외부 DataLoader 전달 필요 | 기존 전처리 모듈에 연결 |
| 모델·손실·학습·복원 함수 | 비교한 원본에서 동일 | 비교한 원본에서 동일 |

이 파일은 모델 개발 초기의 동작 확인 과정을 기록하는 용도로 보관할 수 있습니다. 실제 전체 데이터 학습의 설정과 결과는 `train_full.py` 및 해당 실행 폴더의 기록으로 확인합니다.
