# CSV를 기다리는 동안 준비하는 DenseNet121 모델 코드

대상: 심비대(Cardiomegaly), 폐부종(Edema), 흉수(Pleural Effusion).
PyCharm에서 사용할 수 있는 DenseNet121 + U-multiclass 준비 코드입니다.
전처리와 CSV·이미지 연결은 팀원이 맡은 작업을 그대로 이어받습니다.

## 1. 지금 실행할 수 있는 것

1. ZIP을 풀고 `model_training.py`를 PyCharm 프로젝트에 넣습니다.
2. 프로젝트 인터프리터에 `torch`, `torchvision`을 설치합니다.
   GPU 본 학습용 설치는 아래 공식 안내에서 OS와 CUDA 환경에 맞게 선택합니다.
   https://pytorch.org/get-started/locally/
3. `model_training.py`를 우클릭하여 Run하거나 터미널에서 실행합니다.

```bash
python model_training.py
```

CSV나 실제 이미지가 없어도 실행됩니다. 사전학습 가중치 다운로드도 하지 않습니다.
임시 이미지 2장과 임의 정답을 만들어 입력·출력, 학습 1회, 검증, 저장·복원을 확인합니다.
CPU에서는 시간이 걸릴 수 있습니다.

확인할 형태:

```text
이미지: (2, 3, 224, 224) torch.float32
정답: (2, 3) torch.int64
모델 출력: (2, 3, 3)
동작 확인 성공: 학습 1회, 검증, 저장, 복원, 확률 합 확인
```

손실/F1 수치는 실행마다 환경에 따라 달라질 수 있으며 실제 모델 성능이 아닙니다.
임시 모델은 `runs/smoke_날짜시간/demo_checkpoint.pth`에 저장됩니다.
실제 예측용 불러오기에서는 이 파일을 기본적으로 차단합니다.

## 2. 코드를 읽는 순서

| 함수·클래스 | 역할 |
| --- | --- |
| `CXRModel` | 사전학습 선택, 마지막 출력층 9개로 변경, `[B, 3, 3]` 출력 |
| `check_batch` | 이미지·정답 형식과 라벨 값 확인 |
| `multiclass_loss` | 소견별 CrossEntropyLoss를 동일 비중으로 평균 |
| `run_epoch` | 학습 또는 검증 1회, 손실·F1·혼동행렬 계산 |
| `fit` | 여러 epoch 학습, val loss 기준 최적 모델 저장, 조기 종료 |
| `load_checkpoint` | 모델과 설정 복원 |
| `predict_probabilities` | 클래스 차원에 softmax를 적용해 확률 반환 |
| `smoke_check` | CSV 없는 동작 확인 |

`B`는 한 배치의 이미지 수입니다. 소견 순서는 심비대 → 폐부종 → 흉수입니다.
각 소견의 클래스 순서는 음성 0 → 양성 1 → 불확실 2입니다.
CSV의 원본 `-1`을 Dataset에서 `2`로 변환하고 `torch.long`으로 반환합니다.
여기서 불확실은 판독문 라벨이며, 모델의 낮은 신뢰도와 같은 뜻은 아닙니다.

## 3. 팀원에게 받을 인터페이스

- Dataset·DataLoader 생성 코드와 train/val/test CSV 및 실제 이미지 경로 설정.
- 배치 반환은 `(images, labels)` 두 값으로 통일합니다.
- 이미지: `[B, 3, 224, 224]`, `float32`.
- 정답: `[B, 3]`, `torch.long`, 값은 0/1/2.
- 소견 및 클래스 순서를 위 순서로 고정합니다.
- 비율 유지 패딩·224 크기 조정, 흑백을 3채널로 복제한 뒤 0~1 변환,
  채널 평균 `[0.485, 0.456, 0.406]`, 표준편차 `[0.229, 0.224, 0.225]` 정규화.
- 위 정규화는 Dataset에서 한 번만 적용합니다. 모델 코드에서 다시 적용하지 않습니다.
  이미 정규화된 이미지가 0~1 범위를 벗어나는 것은 정상입니다.
- train만 shuffle하고 val/test는 순서를 유지합니다. 본 학습 배치 크기는 GPU에 맞춥니다.

입력 검사는 형식·유한값만 확인합니다. 정규화 적용 여부, 실제 라벨 연결,
split 중복 여부까지 자동으로 확인하는 기능은 아니므로 팀원의 점검 결과를 받습니다.
이미지 추가 정보까지 반환하는 DataLoader라면 `run_epoch`의 배치 해체 부분을 맞춰야 합니다.

## 4. 실제 데이터가 온 뒤 연결하는 방법

아래는 별도 `train_real.py`에 넣을 연결 예시입니다.
`build_loaders`와 `team_dataset`은 예시 이름이며 현재 제공 파일에는 없습니다.
팀원의 실제 모듈·함수·경로에 맞춰 해당 부분을 바꿉니다.

```python
from model_training import CXRModel, TrainConfig, set_seed, fit, load_checkpoint
# 아래 import는 팀원의 실제 파일/함수 이름으로 교체합니다.
from team_dataset import build_loaders

def main():
    config = TrainConfig(epochs=10, learning_rate=1e-4, patience=3)
    set_seed(config.seed)  # DataLoader와 모델 생성 전에 호출
    train_loader, val_loader, test_loader = build_loaders()
    model = CXRModel(pretrained=True)  # 첫 실행에 ImageNet 가중치 다운로드 필요
    _, best_path = fit(model, train_loader, val_loader, config)
    best_model, checkpoint = load_checkpoint(best_path)
    # 설정 조정은 val에서 수행합니다. 최종 모델 확정 후 test 평가를 별도로 합니다.

if __name__ == "__main__":
    main()
```

Windows/PyCharm에서 DataLoader worker를 사용할 경우에도 위 main 보호문을 유지합니다.
처음에는 실제 train 일부로 시험 학습합니다. 임시 데이터 테스트는
실제 파일 읽기·CSV 연결·학습 가능성·모델 성능을 검증한 것이 아닙니다.

## 5. 저장되는 것과 해석

실제 학습 실행마다 `runs/train_날짜시간/`이 새로 생성됩니다.

- `best_model.pth`: 최소 val loss 모델, optimizer 상태, epoch, 설정, 클래스 순서.
- `config.json`: 입력 규격과 시작 설정.
- `history.csv`: epoch별 학습·검증 손실과 macro F1.
- `best_val_metrics.json`: 최적 epoch의 소견별 클래스 F1과 혼동행렬.

`fit`이 반환하는 model은 마지막 epoch 상태입니다. 최적 모델을 쓸 때는
반환된 `best_path`로 `load_checkpoint`를 실행합니다.
이 버전은 신규 학습용이며 중단 지점부터 재개하는 기능은 아직 구현하지 않았습니다.
선택 기준은 val loss로 고정했고 학습률 자동 조절과 클래스 가중치는 적용하지 않았습니다.
먼저 기준 모델을 학습한 뒤 검증 결과에 따라 설정을 조정합니다.

macro F1은 소견 3개 × 클래스 3개를 같은 비중으로 평균합니다.
관측·예측이 모두 없는 클래스의 F1은 0으로 기록하며, 클래스별 support는
혼동행렬의 행 합으로 확인할 수 있습니다. 클래스 불균형이 있으므로 loss와
전체 F1만 보고 판단하지 말고 소견별 혼동행렬과 클래스별 F1을 함께 확인합니다.
이 준비 코드에는 최종 test 평가, AUROC/PR-AUC, 그래프, 설명 시각화는 포함하지 않습니다.
이진 AUROC/PR-AUC를 추가할 때는 불확실 정답의 평가 처리와 점수 정의를 먼저 정합니다.

## 제공 전 실행 확인

CPU, Python 3.12, torch 2.14.1+cpu, torchvision 0.29.1+cpu 환경에서
임시 데이터 학습 1회·검증·모델 저장·복원과 복원 전후 확률 일치를 확인했습니다.
잘못된 원본 라벨 -1 입력과 임시 모델의 실제 예측용 불러오기도 차단됨을 확인했습니다.
실제 CXR, GPU, 팀원의 CSV·Dataset 연결 및 사전학습 다운로드는 아직 검증하지 않았습니다.

## 참고 자료

- DenseNet121 및 ImageNet 가중치: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.densenet121.html
- CrossEntropyLoss: https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- 모델 저장·복원: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
