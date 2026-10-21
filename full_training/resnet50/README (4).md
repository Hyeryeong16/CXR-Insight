# ResNet50 전체 데이터 학습

심비대(Cardiomegaly), 폐부종(Edema), 흉수(Pleural Effusion)를 분류하기 위해 전체 데이터로 학습한 ResNet50 실험을 정리합니다. DenseNet121과 모델 구조를 비교하고, 불확실 라벨 처리 방식에 따른 학습 결과를 확인하기 위한 실험입니다.

저장 위치: **`model/full_training/resnet50/`**

## 1. 학습 파일

| 파일 | 학습 방식 | 역할 |
|---|---|---|
| `train_full_resnet50.py` | ResNet50 + U-multiclass | 소견별 음성·양성·불확실 3클래스 분류 |
| `train_full_resnet50_uignore.py` | ResNet50 + U-ignore | 소견별 불확실 정답을 제외한 음성·양성 분류 |

## 2. 데이터와 입력

- 데이터셋: MIMIC-CXR-JPG 2.0.0
- 대상 영상: AP·PA 흉부 X-ray, lateral 제외
- 소견 순서: **Cardiomegaly → Edema → Pleural Effusion**

| 분할 | 실제 사용 가능 이미지 수 | 용도 |
|---|---:|---|
| Train | 9,999 | 모델 학습 |
| Validation | 1,077 | 모델 선택·조기 종료·설정 비교 |
| Test | 2,284 | 모델·설정 선택 후 최종 평가 |
| **합계** | **13,360** | |

Train의 원래 대상 10,000장 중 이미지 누락 1장을 제외하여 9,999장으로 학습했습니다. Validation·Test 이미지는 Train에 합치지 않습니다.

| 입력 항목 | 규격 |
|---|---|
| 이미지 텐서 | `[B, 3, 224, 224]` |
| 정답 텐서 | `[B, 3]`, `torch.long` |
| 크기 처리 | 종횡비 유지, 정사각형 패딩, 224×224 크기 조정 |
| 정규화 Mean | `[0.485, 0.456, 0.406]` |
| 정규화 Std | `[0.229, 0.224, 0.225]` |

## 3. 모델과 라벨 처리

| 항목 | U-multiclass | U-ignore |
|---|---|---|
| 모델 | ResNet50 | ResNet50 |
| 학습 클래스 | 음성·양성·불확실 | 음성·양성 |
| 불확실 정답 처리 | 별도 클래스 `2`로 학습 | 해당 소견의 손실·평가에서 제외 |
| 마지막 분류층 출력 수 | 9 | 6 |
| 모델 출력 | `[B, 3, 3]` | `[B, 3, 2]` |
| 예측 확률 | 소견별 클래스 축 softmax | 소견별 클래스 축 softmax |
| Macro F1 구성 | 3개 소견 × 3개 클래스 | 3개 소견 × 2개 클래스 |

U-multiclass의 원본 라벨 매핑은 `0 → 0`(음성), `1 → 1`(양성), `-1 → 2`(불확실)입니다. U-ignore는 불확실을 음성으로 바꾸지 않으며, 한 소견이 불확실해도 같은 이미지의 다른 확정 소견은 사용합니다.

## 4. 학습 설정

| 항목 | 설정 |
|---|---|
| 초기화 | ImageNet 사전학습 가중치 |
| Optimizer | AdamW |
| 초기 학습률 | `1e-4` |
| Weight decay | `1e-4` |
| 최대 epoch | 10 |
| Early stopping | Validation loss 3회 연속 미개선 시 종료 |
| 모델 선택 | 최저 Validation loss |
| Batch size | 4 |
| Seed | 42 |
| DataLoader workers | 0 |

실행별 최종 설정은 해당 실행 폴더의 `config.json`과 학습 소스를 기준으로 확인합니다.

## 5. 실행 방법

### 경로 설정

학습 파일의 프로젝트 루트, 전체 데이터용 CSV 폴더, 이미지 경로를 실행 환경에 맞게 설정합니다. 아래는 이 폴더 위치에서 저장소 루트를 찾는 예시입니다.

```python
from pathlib import Path

# <저장소>/model/full_training/resnet50/<학습파일>.py
PROJECT_DIR = Path(__file__).resolve().parents[3]
CSV_DIR = PROJECT_DIR / "data" / "processed"
IMAGE_ROOT = Path("/home/team30103/physionet_data")
```

실제 train·val·test CSV 위치와 이미지 폴더 구조가 해당 코드의 DataLoader 설정과 일치해야 합니다.

### 서버 실행

저장소 루트에서 실행합니다. `PYTHONPATH`를 설정하여 하위 폴더의 코드가 프로젝트 루트의 공통 모듈을 찾을 수 있도록 합니다.

```bash
conda activate torch210-cu128
cd /home/team30103/chest-xray-multilabel
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

필요한 실험을 선택하여 실행합니다.

```bash
# ResNet50 + U-multiclass
python -u model/full_training/resnet50/train_full_resnet50.py

# ResNet50 + U-ignore
python -u model/full_training/resnet50/train_full_resnet50_uignore.py
```

## 6. 결과 저장·확인

기존 실행 기록의 결과 저장 루트는 다음과 같습니다.

| 실험 | 결과 저장 루트 |
|---|---|
| ResNet50 + U-multiclass | `runs/full_resnet50/` |
| ResNet50 + U-ignore | `runs/full_resnet50_uignore/` |

실행별 결과 폴더에서 다음 자료를 함께 관리합니다. 실제 생성 파일은 해당 실행본을 기준으로 확인합니다.

| 파일 | 내용 |
|---|---|
| `best_model.pth` | 최저 Validation loss로 선택한 체크포인트 |
| `config.json` | 모델·라벨 처리·입력·학습 설정 |
| `history.csv` | epoch별 학습·검증 loss와 F1 기록 |
| `best_val_metrics.json` | 선택 모델의 Validation 지표 |

평가·추론에는 선택된 `best_model.pth`를 복원합니다. 마지막 epoch의 결과와 선택 모델의 결과를 구분합니다.

## 7. 전체 데이터 학습 결과

기존 공유된 학습 기록에 정리된 **선택 모델의 Validation 결과**입니다. 실행별 원본 지표는 `best_val_metrics.json` 또는 체크포인트의 `val_metrics`를 기준으로 확인합니다.

| 학습 방식 | 선택 epoch | 종료 epoch | Val loss | Macro F1 | 양성 F1 평균 |
|---|---:|---:|---:|---:|---:|
| ResNet50 + U-multiclass | 4 | 7 | 0.7297 | 0.4434 — 3클래스 | 미기재 |
| ResNet50 + U-ignore | 2 | 5 | 0.5265 | 0.6895 — 2클래스 | 0.5944 |

**위 표는 Test 평가 결과가 아닙니다.** AUROC·소견별 세부 지표는 해당 실행 결과가 확인된 항목만 추가합니다.

## 8. 지표 해석과 비교 기준

- **U-multiclass Macro F1:** 세 소견 × 세 클래스의 F1, 총 9개 점수의 평균입니다.
- **U-ignore Macro F1:** 소견별 불확실 정답을 제외한 뒤, 세 소견 × 두 클래스의 F1, 총 6개 점수의 평균입니다.
- **양성 F1 평균:** 심비대·폐부종·흉수의 양성 F1 세 개를 평균합니다.
- **U-multiclass와 U-ignore의 원래 loss·Macro F1을 직접 비교해 우열을 정하지 않습니다.** 학습·평가 대상이 다릅니다.
- DenseNet과 ResNet의 성능을 비교할 때는 같은 Validation split, 확정 라벨 마스크, 확률 계산 방식, 판정 규칙을 적용하여 양성 F1·Recall·Precision·AUROC 등을 확인합니다.
- 모델·설정 선택에는 Validation을 사용하고, 선택이 끝난 뒤 Test에서 최종 성능을 평가합니다.

ResNet50은 DenseNet121과 비교하기 위한 구조 후보로 학습했습니다. 이 결과만으로 특정 모델이나 불확실 처리 방식이 항상 우월하다고 결론 내리지는 않습니다.
