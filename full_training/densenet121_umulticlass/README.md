# DenseNet121 U-multiclass 전체 데이터 학습

MIMIC-CXR의 심비대, 폐부종, 흉수를 각각 음성·양성·불확실의 3클래스로 예측하는 학습 코드입니다. 기존 파일럿 실험 이후 전체 데이터 학습을 위해 준비한 기본·Weight decay·증강·가중치·Scheduler 실험을 정리합니다.

## 실험 구성

| 파일 | 실험 목적 | 설정 확인 범위 |
|---|---|---|
| `train_full.py` | 기본 모델의 비교 기준 마련 | 원본 코드 확인 |
| `train_full_wd.py` | Weight decay 변경을 통한 과적합 완화 실험 | 서버 파일 존재 확인; 구체적 변경값은 해당 코드 기준 |
| `train_full_augmented.py` | 데이터 증강을 통한 일반화 성능 개선 실험 | 서버 파일 존재 확인; 증강 종류·확률·강도는 해당 코드 기준 |
| `train_full_weighted.py` | 클래스 불균형 대응 | 원본 코드 확인 |
| `train_full_weighted_scheduler.py` | 가중치 실험에 학습률 조절 추가 | 원본 코드 확인 |

표의 나열은 설명 순서입니다. 아래의 구체적인 공통 설정·손실·지표·저장 경로는 원본을 확인한 기본·가중치·Scheduler 3개 파일을 기준으로 합니다. `train_full_wd.py`와 `train_full_augmented.py`의 변경값과 성능 결과는 아직 확인하지 않았습니다.

### 원본에서 확인한 손실·Scheduler 차이

| 순서 | 파일 | 훈련 손실 | 검증 손실 | Scheduler | 추가 보고 |
|---|---|---|---|---|---|
| 1 | `train_full.py` | 가중치 없는 Cross Entropy | 가중치 없는 Cross Entropy | 없음 | 3클래스 Macro F1 |
| 2 | `train_full_weighted.py` | 클래스 가중치 Cross Entropy | 가중치 없는 Cross Entropy | 없음 | 불확실 클래스 F1 |
| 3 | `train_full_weighted_scheduler.py` | 2번과 동일 | 가중치 없는 Cross Entropy | ReduceLROnPlateau | 불확실 F1, 확정 라벨 AUROC, 학습률 |

기본·가중치·Scheduler 원본 Python 파일을 확인했습니다. Scheduler 파일의 `source_sha256`이 첨부한 `train_full_weighted.py`의 해시와 일치합니다.

### Weight decay 변경 실험

`train_full_wd.py`는 Weight decay 설정을 바꾸어 기본 모델과 비교하는 실험 항목입니다. Weight decay는 학습 중 가중치가 지나치게 커지는 것을 억제하는 정규화 설정입니다. 실험 목적은 과적합 완화이며, 실제 개선 여부는 검증 결과로 판단합니다. 변경한 수치는 해당 파일 또는 실행 폴더의 `config.json`으로 확인합니다.

### 데이터 증강 실험

`train_full_augmented.py`는 학습 이미지에 변형을 적용하는 데이터 증강 실험 항목입니다. 실험 목적은 제한된 학습 데이터에서 일반화 성능을 개선하는 것입니다. 실제 적용한 증강 방식·확률·강도와 검증·테스트 변환은 해당 파일 및 연결된 전처리 코드로 확인합니다. 성능 향상 여부는 실험 결과를 바탕으로 판단합니다.

## 기본·가중치·Scheduler 3개 파일의 공통 모델과 설정

| 항목 | 코드에서 확인한 설정 |
|---|---|
| Backbone | DenseNet121, ImageNet 사전학습 가중치 |
| 학습 범위 | 모델 전체 파라미터를 AdamW에 전달 |
| 입력 | 3채널 224 × 224 |
| 출력 | `[배치, 소견 3개, 클래스 3개]` |
| 소견 순서 | Cardiomegaly → Edema → Pleural Effusion |
| 클래스 순서 | 음성 0 → 양성 1 → 불확실 2 |
| 원본 라벨 매핑 | 0→0, 1→1, -1→2 |
| Optimizer | AdamW |
| 초기 학습률 | 0.0001 |
| Weight decay | 0.0001 |
| Batch size | 4 |
| 최대 Epoch | 10 |
| Early stopping | 검증 손실이 연속 3회 개선되지 않으면 종료 |
| Seed | 42 |
| 최적 모델 선택 기준 | 검증 손실 최소 |

세 소견별 Cross Entropy를 계산해 평균합니다. 불확실은 독립된 세 번째 클래스로 손실과 Macro F1에 포함합니다. U-ignore처럼 불확실 라벨을 손실에서 제외하는 코드가 아닙니다.

가중치 실험은 학습 데이터에서 소견별 음성·양성·불확실 빈도를 구한 뒤 `1 / sqrt(빈도)`를 계산하고, 소견별 평균 가중치가 1이 되도록 정규화합니다. 검증 손실은 가중치를 사용하지 않습니다. 코드에서 학습 데이터가 9,999장인지 검사하고, 학습 데이터에 빠진 클래스가 있으면 오류를 냅니다.

Scheduler는 검증 손실을 모니터링하며, 연속 2회 개선되지 않으면 다음 epoch의 학습률을 절반으로 줄입니다. 설정은 `patience=1`, `factor=0.5`, `min_lr=0.000001`입니다. 학습률 감소 시 early stopping 카운터를 초기화하지 않습니다.

## 지표 해석

- **Macro F1**: 소견별 3클래스 F1을 구한 후 전체 9개 값을 평균합니다. 관측·예측이 모두 없는 클래스의 F1은 0으로 기록합니다.
- **불확실 F1**: 각 소견의 클래스 2 F1입니다. 가중치 및 Scheduler 코드가 출력합니다.
- **AUROC**: Scheduler 코드에서만 계산합니다. 소견별 원본 0/1 확정 라벨만 평가하고, 불확실 라벨 2는 AUROC 계산에서 제외합니다. 점수는 3클래스 softmax 중 양성 클래스 확률이며 음성·양성만으로 재정규화하지 않습니다.
- 세 소견 모두 AUROC를 계산할 수 있을 때 평균 AUROC를 보고합니다. 한 소견의 확정 라벨이 한 클래스뿐이면 해당 AUROC와 평균은 미정의로 둡니다.
- AUROC는 보고용이며 체크포인트 선택은 검증 손실로 수행합니다.
- 테스트 DataLoader는 생성하지만 학습·검증 함수에 넣지 않습니다. 별도 테스트 평가가 필요합니다.

U-ignore의 2클래스 Macro F1과 여기의 3클래스 Macro F1은 집계 대상이 다릅니다.

## 프로젝트 경로와 기존 의존 파일

원본을 확인한 기본·가중치·Scheduler 코드에 다음 서버 경로가 직접 지정되어 있습니다.

| 항목 | 현재 설정 |
|---|---|
| 프로젝트 | `/home/team30103/chest-xray-multilabel` |
| CSV 폴더 | 프로젝트의 `data/processed/` |
| 이미지 폴더 | `/home/team30103/physionet_data` |
| 이미지 배치 방식 | `image_layout="flat"` |
| DataLoader 키 | `train`, `validate`, `test` |

기본·가중치·Scheduler 코드는 기존 프로젝트 최상위의 **`cxr_preprocessing.py`가 필요합니다.** 학습 코드와 기존 전처리 모듈을 함께 사용해야 합니다. 필요한 실행 환경은 기존 서버의 Python·PyTorch·torchvision 환경과 전처리 모듈 의존 패키지입니다. 가중치 코드는 pandas, Scheduler 코드는 scikit-learn도 사용합니다. 세 스크립트는 CUDA가 없으면 실행을 중단합니다.

기본·가중치·Scheduler 세 파일에는 `TRAIN_CSV = ... / "train_final.csv"`, `VAL_CSV = ... / "val.csv"`, `TEST_CSV = ... / "test.csv"`가 선언되어 있습니다. 그러나 실제 DataLoader 호출에는 이 개별 파일 경로를 전달하지 않고 `csv_dir=CSV_DIR`만 전달합니다. **실제로 읽는 CSV 파일명과 전처리는 기존 `cxr_preprocessing.py`의 구현이 결정합니다.** 이 세 파일만으로 실제 학습에 사용한 모든 CSV와 전체 데이터 수를 확정할 수는 없습니다.

다른 서버에서 재실행할 때는 각 파일의 `PROJECT_DIR`, 이미지 경로, 이미지 배치 방식 및 기존 전처리 모듈의 CSV 선택 규칙을 실제 환경에 맞춥니다. 문서화 과정에서는 학습 이력을 보존하기 위해 원본 경로를 바꾸지 않았습니다.

## 저장 위치

| 파일 | 현재 결과 저장 경로 |
|---|---|
| `train_full.py` | `runs/pilot/train_.../` |
| `train_full_weighted.py` | `runs/full_weighted/train_.../` |
| `train_full_weighted_scheduler.py` | `runs/full_weighted_scheduler/train_.../` |
| `train_full_wd.py` | 해당 파일의 `OUTPUT_ROOT` 또는 실행 설정 확인 필요 |
| `train_full_augmented.py` | 해당 파일의 `OUTPUT_ROOT` 또는 실행 설정 확인 필요 |

**기본 코드의 저장 폴더 이름은 `pilot`로 남아 있습니다.** 주석과 출력 문구에도 pilot·시험 학습 표현이 남아 있습니다. 이 이름만으로 파일럿 데이터로 학습했다고 판단하면 안 됩니다. 데이터 규모는 실행 당시의 DataLoader 출력과 CSV로 확인합니다.

확인한 기본·가중치·Scheduler 각 실행 폴더에는 `best_model.pth`, `config.json`, `history.csv`, `best_val_metrics.json`이 저장됩니다. Scheduler 코드는 체크포인트에 Scheduler 상태와 학습률도 저장합니다.

## 실행 예시

저장소에서 이 문서와 실험 코드 5개를 `full_training/densenet121_umulticlass/`에 두는 구성을 기준으로 합니다. Weight decay·증강 파일은 별도로 업로드합니다. `PROJECT_DIR`이 기존 서버 프로젝트 경로이고, 해당 경로의 전처리 모듈과 데이터가 준비된 상태에서 실행합니다.

```bash
python -u full_training/densenet121_umulticlass/train_full.py
python -u full_training/densenet121_umulticlass/train_full_weighted.py
python -u full_training/densenet121_umulticlass/train_full_weighted_scheduler.py
```

Weight decay·증강 파일을 같은 폴더에 업로드하고 해당 파일의 경로와 설정을 확인한 뒤에는 다음과 같이 실행할 수 있습니다.

```bash
python -u full_training/densenet121_umulticlass/train_full_wd.py
python -u full_training/densenet121_umulticlass/train_full_augmented.py
```

위 명령은 학습 실행용입니다. GitHub에 기록만 올리려면 재학습할 필요는 없습니다. 확인한 기본·가중치·Scheduler 세 파일은 학습 시작 부분이 최상위에 있으므로 **import해도 데이터 연결과 학습이 실행됩니다.** 모델 클래스만 가져오기 위한 모듈로 직접 import하지 마세요.

## 검증 및 결과 범위

업로드 정리 과정에서 다음을 확인했습니다.

- 기본·가중치·Scheduler 원본 3개 파일의 Python 문법과 기존 업로드 묶음 내 원본 일치 여부.
- 사용자가 제공한 서버 파일 목록에서 Weight decay·증강 파일의 존재.
- Scheduler가 기록한 원본 SHA-256과 첨부 가중치 파일의 일치.
- 가중치 코드와 Scheduler 코드의 클래스 가중치 계산 및 손실 수식이 동일함.
- Scheduler의 AUROC 함수만 분리해, 함수 내 사전 검사와 같은 예제로 계산 결과를 확인함.

정리 환경에는 PyTorch·CUDA·이미지·CSV 및 실제 `cxr_preprocessing.py`가 없어 전체 학습과 Scheduler 사전 검사는 실행하지 않았습니다. 실제 실행 로그와 체크포인트도 이번 첨부에는 포함되지 않았으므로, 이 문서는 코드 차이를 정리하며 실험 성능의 우열을 확정하지 않습니다.

확인한 기본·가중치·Scheduler 코드는 모두 최대 10 epoch + early stopping 버전입니다. Weight decay·증강 파일의 설정은 별도 확인 대상입니다. 이 문서에는 50 epoch 또는 early stopping 해제 실험 결과를 포함하지 않습니다.
