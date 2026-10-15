# DenseNet121 파일럿 학습 및 과적합 비교 실험

MIMIC-CXR-JPG 흉부 X-ray에서 **Cardiomegaly(심비대), Edema(폐부종), Pleural Effusion(흉수)**을 분류하는 파일럿 학습 코드입니다.

각 소견을 **음성·양성·불확실(Uncertain)의 3개 클래스**로 분류하는 U-multiclass 방식을 사용합니다. 전체 이미지 다운로드를 기다리는 동안 총 1,000장의 이미지로 데이터 연결, 학습, 모델 저장 및 과적합 관련 실험을 진행했습니다.

이 문서는 2026-10-15까지의 서버 실행 기록과 공유된 코드를 기준으로 작성했습니다. 아래 성능은 모두 **파일럿 Validation 결과**이며, 최종 데이터나 Test 평가 결과가 아닙니다.

## 1. 파일 구성

GitHub에는 학습 스크립트를 `pilot_training/` 폴더에 정리합니다.

| 파일 | 역할 | 학습 범위·변경 내용 |
| --- | --- | --- |
| `train_pilot.py` | 기본 비교 모델 | DenseNet121 전체층 학습, 일반 CrossEntropy |
| `train_pilot_frozen.py` | 분류층만 학습 | 특징 추출 부분 고정, classifier만 학습 |
| `train_pilot_weighted.py` | 클래스 불균형 대응 | Train 분포로 계산한 Weighted CrossEntropy 사용 |
| `train_pilot_partial.py` | 일부 층 학습 | `denseblock4`, `norm5`, `classifier` 학습 |
| `train_pilot_augmented.py` | 데이터 증강 실험 | 전체층 학습 + Train에만 약한 회전·이동 |

공통 전처리 모듈인 **`cxr_preprocessing.py`가 별도로 필요**합니다. 현재 서버에서는 프로젝트 최상위에 있습니다. GitHub에서 내려받아 실행할 때도 이 파일을 프로젝트 최상위에 준비해야 합니다.

`.py` 파일을 수정해도 기존 모델 준비 노트북(`.ipynb`)에는 자동으로 반영되지 않습니다. 이 문서의 실험 결과는 위 `.py` 스크립트로 실행한 결과입니다.

## 2. 데이터와 라벨

### 파일럿 분할

| 구분 | 이미지 수 | CSV 파일명 | 용도 |
| --- | ---: | --- | --- |
| Train | 800 | `pilot_train.csv` | 모델 학습 |
| Validation | 100 | `pilot_val.csv` | epoch 선택, 설정 비교, Early stopping |
| Test | 100 | `pilot_test.csv` | 최종 평가용으로 보관; 현재 성능 평가에는 미사용 |

분할 간 `subject_id`, `study_id`, `dicom_id` 중복을 검사합니다. 파일럿 CSV는 공식 Train에서 다시 나눈 목록이므로 원래 `split` 값이 모두 `train`이었습니다. 전처리 코드는 `pilot_` 파일에 한해 원래 값을 `source_split`으로 보존하고, 파일에 따라 메모리상의 `split`을 `train`·`validate`·`test`로 설정합니다. CSV 원본은 변경하지 않습니다.

### 필수 CSV 컬럼

```text
subject_id,study_id,dicom_id,ViewPosition,split,Cardiomegaly,Edema,Pleural Effusion
```

- 촬영 방향은 `AP` 또는 `PA`를 사용합니다.
- CSV의 소견 라벨은 원본 값 `0`, `1`, `-1`을 유지합니다.
- 빈 라벨, 허용 범위 밖의 라벨, 중복 이미지 식별자 등을 확인합니다.
- 이미지 경로는 CSV 식별자와 서버의 이미지 폴더를 이용해 생성합니다.

### 라벨 순서와 변환

소견 순서: **Cardiomegaly → Edema → Pleural Effusion**

| 의미 | CSV 원본 라벨 | 학습 클래스 번호 |
| --- | ---: | ---: |
| 음성 / negative | 0 | 0 |
| 양성 / positive | 1 | 1 |
| 불확실 / uncertain | -1 | 2 |

`-1 → 2`는 CrossEntropy에 맞추기 위한 클래스 번호 변환입니다. `2`가 더 심한 상태라는 뜻은 아닙니다. Uncertain은 데이터에 부여된 정답 클래스이며, 모델이 계산한 확신도와는 구분합니다.

## 3. 전처리와 모델 입력

`cxr_preprocessing.py`가 이미지마다 다음 처리를 수행합니다.

1. JPG를 흑백으로 읽습니다.
2. 검은 패딩을 넣어 정사각형을 만듭니다.
3. 가로·세로 비율을 유지한 상태에서 `224 × 224`로 조정합니다.
4. 픽셀을 `0~1` 범위의 `float32`로 변환합니다.
5. 같은 흑백 값을 3개 채널에 복제합니다.
6. ImageNet 평균·표준편차로 정규화합니다.

```python
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
```

| 데이터 | 배치 형태 | 자료형 |
| --- | --- | --- |
| 이미지 | `[B, 3, 224, 224]` | `torch.float32` |
| 정답 | `[B, 3]` | `torch.int64` / `torch.long` |
| 모델 출력 | `[B, 3, 3]` | logits |

모델 출력의 두 번째 축은 소견, 세 번째 축은 음성·양성·불확실 클래스입니다. 기본 손실 함수는 **소견별 3클래스 CrossEntropy를 계산한 뒤 평균**합니다. 학습 손실 계산 전에는 softmax를 적용하지 않습니다. 예측 확률을 구할 때 클래스 축에 softmax를 적용합니다.

전처리와 배치 구성은 CPU에서 수행하고, 학습 직전에 이미지·라벨 배치를 GPU로 이동합니다. 원본 JPG를 수정하거나 전처리 이미지를 별도 파일로 저장하는 방식이 아니라, DataLoader가 읽을 때 전처리합니다.

## 4. 서버 환경과 경로

현재 코드에는 팀 서버의 **절대 경로**가 들어 있습니다. 다른 서버나 PC에서 실행하려면 경로와 GPU 환경을 먼저 맞춰야 합니다.

| 항목 | 현재 서버 설정 |
| --- | --- |
| 프로젝트 | `/home/team30103/chest-xray-multilabel` |
| 전처리 모듈 | `/home/team30103/chest-xray-multilabel/cxr_preprocessing.py` |
| 파일럿 CSV 폴더 | `/home/team30103/chest-xray-multilabel/data/processed/pilot` |
| 이미지 폴더 | `/home/team30103/physionet_data` |
| 이미지 구조 | `flat`: `이미지 폴더/dicom_id.jpg` |
| Conda 환경명 | `torch210-cu128` |

사용 라이브러리: `torch`, `torchvision`, `numpy`, `pandas`, `Pillow`, `matplotlib`.

전처리 모듈의 파일럿 CSV 이름 설정은 다음과 같아야 합니다.

```python
CSV_NAMES = {
    "train": "pilot_train.csv",
    "validate": "pilot_val.csv",
    "test": "pilot_test.csv",
}
```

모든 CSV 행에 대응하는 JPG가 있어야 전체 DataLoader를 만들 수 있습니다. 파일럿 CSV 이름 설정은 `cxr_preprocessing.py`의 전역 설정이므로, 여러 실험을 실행하는 동안 동일하게 유지합니다.

## 5. 실행 방법

### 환경 활성화와 GPU 확인

아래 명령은 **GPU 서버에 연결된 터미널**에서 실행합니다.

```bash
source /home/team30103/miniconda3/etc/profile.d/conda.sh
conda activate torch210-cu128
cd /home/team30103/chest-xray-multilabel
python -c "import torch; print('torch:', torch.__version__); print('GPU:', torch.cuda.is_available())"
```

학습 스크립트는 GPU를 인식하지 못하면 중단하도록 설정되어 있습니다.

### 실행할 실험 선택

현재 서버에서는 학습 파일을 프로젝트 최상위에 두고 실행했습니다. 아래에서 필요한 명령 하나를 선택합니다.

| 실험 | 현재 서버 실행 명령 |
| --- | --- |
| 기본 전체층 | `python train_pilot.py` |
| 분류층만 | `python train_pilot_frozen.py` |
| 클래스 가중치 | `python train_pilot_weighted.py` |
| 일부 층 | `python train_pilot_partial.py` |
| 약한 증강 | `python train_pilot_augmented.py` |

GitHub와 같이 스크립트를 `pilot_training/` 안에 배치했다면 프로젝트 최상위에서 다음처럼 실행합니다.

```bash
python pilot_training/train_pilot.py
```

나머지 실험도 파일명 앞에 `pilot_training/`을 붙이면 됩니다. 스크립트의 `PROJECT_DIR`은 전처리 모듈과 CSV가 있는 실제 프로젝트 경로를 가리켜야 합니다.

### 공통 학습 설정

| 항목 | 설정 |
| --- | --- |
| 초기화 | ImageNet 사전학습 가중치 |
| 최대 epoch | 10 |
| Batch size | 4 |
| DataLoader workers | 0 |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Weight decay | `1e-4` |
| Early stopping patience | 3 |
| Seed | 42 |
| epoch 선택 기준 | 최저 Validation loss |

각 스크립트는 ImageNet 사전학습 모델에서 **새 실험을 시작**합니다. 저장된 이전 학습을 이어서 진행하는 재개 코드가 아닙니다.

epoch를 수정할 때는 파일 아래쪽의 **실제 실행부 `real_config = TrainConfig(...)`**를 확인합니다. dataclass의 기본값보다 실행부에서 명시한 값이 우선합니다.

```python
real_config = TrainConfig(
    epochs=10,
    learning_rate=1e-4,
    weight_decay=1e-4,
    patience=3,
    output_root=str(OUTPUT_ROOT),
)
```

Validation loss가 3 epoch 연속 개선되지 않으면 종료합니다. 최대 10 epoch로 설정해도 더 일찍 끝날 수 있습니다. 실제 학습 실행 여부는 `RUN_REAL_TRAINING = True`로 설정합니다.

## 6. 실험별 추가 설정

### 분류층만 학습

- `backbone.features`의 파라미터를 고정합니다.
- 고정된 특징 추출 부분은 학습 중에도 `eval()`로 유지하여 BatchNorm 통계 업데이트를 막습니다.
- 학습 대상은 `backbone.classifier.weight`, `backbone.classifier.bias`입니다.
- 학습할 파라미터 수: **9,225개**.

### 일부 층 학습

- 특징 추출 부분을 고정한 뒤 `denseblock4`, `norm5`를 학습 가능하게 설정합니다.
- `classifier`도 함께 학습합니다.
- 앞부분의 고정된 층은 `eval()` 상태로 유지합니다.
- 학습할 파라미터 수: **2,169,353개**.

### 클래스 가중치 적용

가중치는 **Train 정답만 사용해 소견별로 계산**합니다. Validation·Test 분포를 가중치 계산에 사용하지 않습니다.

```text
소견별 클래스 가중치 = 해당 소견의 Train 총 개수 / (3 × 해당 클래스 개수)
```

| 소견 | Train 개수: 음성 / 양성 / 불확실 | 가중치: 음성 / 양성 / 불확실 |
| --- | --- | --- |
| Cardiomegaly | 454 / 300 / 46 | 0.587 / 0.889 / 5.797 |
| Edema | 483 / 211 / 106 | 0.552 / 1.264 / 2.516 |
| Pleural Effusion | 383 / 377 / 40 | 0.696 / 0.707 / 6.667 |

학습 시에는 가중치를 적용해 소견별 샘플 손실의 평균을 구하고, 세 소견의 손실을 평균합니다. 가중치 모델의 화면에 표시되는 `train loss`·`val loss`는 비교를 위한 **비가중치 CrossEntropy**입니다. 실제 가중치 학습 손실은 별도로 기록하며, Validation loss와 Early stopping에도 클래스 가중치를 적용하지 않습니다.

### 약한 데이터 증강

- Train에만 무작위 회전 **±3°**, 가로·세로 이동 **최대 2%**를 적용합니다.
- Bilinear 보간과 검은색 빈 영역 채우기를 사용합니다.
- 좌우·상하 반전은 적용하지 않습니다.
- 기존 전처리 결과의 정규화를 되돌린 뒤 증강하고, 같은 평균·표준편차로 다시 정규화합니다.
- Validation·Test에는 증강을 적용하지 않습니다.
- 원본 JPG는 변경하지 않습니다.

## 7. 결과 파일과 평가 기준

| 실험 | 결과 저장 루트 |
| --- | --- |
| 기본 모델 | `runs/pilot/` |
| 분류층만 | `runs/pilot/frozen/` |
| 클래스 가중치 | `runs/pilot/weighted/` |
| 일부 층 | `runs/pilot/partial/` |
| 약한 증강 | `runs/pilot/augmented/` |

실행마다 저장 루트 아래에 `train_날짜_시간/` 폴더가 만들어집니다.

| 파일 | 내용 |
| --- | --- |
| `best_model.pth` | Validation loss가 가장 낮았던 모델과 학습 메타데이터 |
| `config.json` | 학습 설정, 소견·클래스 순서, 입력 규격 등 |
| `history.csv` | epoch별 Train·Validation loss와 Macro-F1 |
| `best_val_metrics.json` | 선택된 모델의 Validation loss, F1, Confusion Matrix 등 |

**마지막 epoch 모델과 최적 모델은 다를 수 있습니다.** 스크립트는 학습 종료 후 저장된 최적 모델을 다시 불러오는 것까지 확인합니다. Validation loss와 Macro-F1의 최적 시점도 서로 다를 수 있으므로, 로그의 최대 F1을 저장 모델의 F1로 해석하지 않습니다.

Macro-F1은 **3개 소견 × 3개 클래스의 F1, 총 9개 값을 동일한 비중으로 평균**한 값입니다. 정확도와는 다른 지표입니다. 정답과 예측이 모두 없는 클래스의 F1은 현재 코드에서 0으로 기록합니다.

Confusion Matrix는 소견마다 3×3으로 기록합니다.

- 행: 실제 정답
- 열: 모델 예측
- 행·열 순서: **음성 → 양성 → 불확실**

## 8. 파일럿 실험 결과

아래는 각 실험에서 **Validation loss가 가장 낮았던 checkpoint**의 결과입니다. 더 높은 Macro-F1이 다른 epoch에 있더라도 아래 표에는 선택된 checkpoint의 값을 사용했습니다.

| 실험 | 선택된 epoch | Validation loss ↓ | Validation Macro-F1 ↑ |
| --- | ---: | ---: | ---: |
| 기본 전체층 학습 | 3 | **0.8464** | **0.4110** |
| Weight decay 강화 (`1e-3`) | 1 | 0.8533 | 0.3865 |
| 분류층만 학습 | 9 | 0.8585 | 0.3839 |
| 클래스 가중치 적용 | 5 | 1.0455 | 0.3975 |
| 일부 층 학습 | 2 | 0.8868 | 0.3661 |
| 약한 데이터 증강 | 3 | 0.8516 | 0.3827 |

Weight decay 강화는 기본 모델의 설정을 바꿔 진행한 별도 실험이며, 위 파일 목록에 전용 스크립트는 없습니다. 공유된 기본 스크립트는 `weight_decay=1e-4`입니다.

### Uncertain 성능과 다른 클래스의 변화

| 소견 | 기본 모델 Uncertain 재현율 | 가중치 모델 Uncertain 재현율 | 가중치 모델 Uncertain Precision / F1 |
| --- | ---: | ---: | --- |
| Cardiomegaly | 0% | 12.5% | 0.091 / 0.105 |
| Edema | 0% | 44.4% | 0.143 / 0.216 |
| Pleural Effusion | 0% | 20.0% | 0.143 / 0.167 |

가중치 적용으로 Uncertain을 맞히는 사례는 늘었지만, 다른 클래스의 성능 저하도 나타났습니다. 특히 **Edema 음성 F1은 0.813 → 0.265**로 감소했습니다. 따라서 기본 모델은 전체 지표의 비교 기준으로 유지하고, 가중치 모델은 Uncertain 성능을 개선하기 위한 후보로 비교합니다.

### 파일럿 결론

- Train loss 감소와 Validation loss 상승으로 과적합 양상을 확인했습니다.
- 규제 강화, 학습 범위 제한, 약한 증강으로 기존 모델보다 뚜렷한 검증 성능 개선을 확인하지 못했습니다.
- 가중치 적용은 클래스 불균형에 대응하는 실험이며, 과적합 자체를 해결하지는 못했습니다.
- 기본 모델의 전체 Macro-F1이 가장 높아도 Uncertain 재현율이 0%이므로, 최종 목표를 충분히 달성한 모델이라고 보지는 않습니다.
- Validation이 100장이고 Uncertain이 소견별 5~9장이므로 결과 해석에 한계가 있습니다. 여러 번 설정을 비교한 결과도 파일럿 탐색 결과로 취급합니다.

## 9. 전체 데이터로 전환할 때

전체 이미지 다운로드 완료 후 **Train 10,000장 / Validation 1,077장**으로 기본 모델과 가중치 모델을 다시 비교할 계획입니다. 최종 CSV에서 Train·Validation 간 환자·검사·이미지 중복 0개를 확인했으며, 실제 JPG의 존재와 읽기 가능 여부는 별도로 확인해야 합니다.

| 확인 항목 | 필요한 작업 |
| --- | --- |
| 학습 파일 | 파일럿 스크립트를 보존하고 전체 데이터용 실행 파일을 별도로 준비 |
| CSV 폴더 | 학습 스크립트의 `CSV_DIR`을 `data/processed`로 변경 |
| CSV 경로 변수 | `TRAIN_CSV`, `VAL_CSV`, `TEST_CSV`를 전체 CSV에 맞게 변경 |
| CSV 파일명 매핑 | 전처리 모듈의 `CSV_NAMES`를 `train.csv`, `val.csv`, `test.csv`로 변경 |
| 이미지 폴더 | 전체 CSV의 `dicom_id`와 실제 이미지 경로 확인 |
| 결과 폴더 | 파일럿과 구분되는 별도 `OUTPUT_ROOT` 사용 |
| 클래스 가중치 | 전체 Train 분포로 다시 계산 |
| 비교 조건 | 동일 분할·초기화·학습 설정에서 가중치 적용 여부 비교 |
| 평가 | 전체 Macro-F1, 클래스별 Precision·Recall·F1, Confusion Matrix 비교 |

전체 데이터용 파일을 준비할 때 필요한 CSV 매핑은 다음과 같습니다.

```python
CSV_NAMES = {
    "train": "train.csv",
    "validate": "val.csv",
    "test": "test.csv",
}
```

공유 전처리 모듈의 `CSV_NAMES`를 바꾸면 파일럿 스크립트에도 영향을 줍니다. 전환 중 파일럿과 전체 데이터 실험을 같은 모듈 설정으로 동시에 실행하지 않도록 실행 구성을 구분합니다.

**최종 모델은 전체 Macro-F1뿐 아니라 Uncertain 인식과 음성·양성 성능의 균형을 보고 Validation으로 선택합니다. Test는 모델·설정을 확정한 뒤 최종 성능 평가에 사용합니다.**

## 10. 자주 발생한 오류

| 증상 | 확인·해결 방법 |
| --- | --- |
| `ModuleNotFoundError: No module named 'torch'` | 새 터미널에서 `torch210-cu128` 환경을 활성화했는지 확인 |
| `ModuleNotFoundError: No module named 'cxr_preprocessing'` | `PROJECT_DIR`과 프로젝트 최상위의 전처리 모듈 위치 확인 |
| `최종 선정 CSV가 없습니다` | `CSV_DIR`과 전처리 모듈의 `CSV_NAMES` 확인 |
| `split 값이 'validate'과 일치하지 않습니다` | 파일럿 파일의 원래 split과 파일별 분할 처리 확인; 공식 CSV 검사를 무조건 제거하지 않음 |
| `이미지 파일이 없습니다` | 이미지 다운로드 완료 여부, `IMAGE_ROOT`, `flat` 구조와 dicom_id 확인 |
| `GPU가 인식되지 않았습니다` | Conda 환경과 `torch.cuda.is_available()` 확인 |
| NumPy 배열이 쓰기 불가능하다는 경고 | 라벨 배열의 `.to_numpy(dtype=np.int64).copy()` 적용 확인 |

현재 실행 경로와 설정이 각 스크립트에 들어 있으므로, 다른 팀원이 실행할 때는 경로와 CSV 매핑을 먼저 확인합니다.
