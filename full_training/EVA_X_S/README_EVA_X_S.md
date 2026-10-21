# EVA-X-S 기반 흉부 X-ray 3개 소견 분류

MIMIC-CXR-JPG 영상에서 **심비대(Cardiomegaly), 폐부종(Edema), 흉수(Pleural Effusion)**를 분류하는 전체 데이터 학습 실험입니다. DenseNet121의 비교 모델로 EVA-X-S를 적용했습니다.

저장소 내 예정 위치: `modle/full_training/EVA_X_S/`

> 이 README는 전체 데이터 학습 결과를 기록합니다. Pilot 실험은 포함하지 않으며, 아래 성능은 모두 **검증셋(validation)** 결과입니다.

## 1. 실험 개요

| 항목 | 내용 |
|---|---|
| 모델 | EVA-X-S (`eva_x_small_patch16`) |
| 초기 가중치 | 공개 CXR 자기지도 사전학습 가중치 |
| 우리 팀의 학습 | 소견 라벨을 사용한 모델 전체 지도 미세조정 |
| 대상 소견 | Cardiomegaly / Edema / Pleural Effusion |
| 라벨 처리 | U-ignore |
| 출력 | `[batch_size, 3, 2]` — 각 소견의 음성·양성 로짓 |
| 실험 목적 | CXR 사전학습 모델 적용 시 기존 DenseNet121 대비 검증 성능 변화 확인 |

**우리 팀이 EVA-X의 자기지도 사전학습을 직접 수행한 것은 아닙니다.** 연구진이 학습한 공개 가중치를 가져와 우리 라벨로 미세조정했습니다. 이 실험에서는 교사 모델, 패치 마스킹, 자기지도 추가 사전학습을 사용하지 않았습니다.

## 2. 파일 및 의존 관계

| 파일·경로 | 역할 / 배치 위치 |
|---|---|
| `modle/full_training/EVA_X_S/train_evax_s_uignore_full.py` | EVA-X-S 전체 학습 코드. 기존 DenseNet 학습 코드에서 모델·정규화·메타데이터·결과 경로를 변경하여 생성 |
| `modle/full_training/EVA_X_S/README.md` | 실험 설정, 실행 방법, 결과 정리 |
| `cxr_preprocessing.py` | 저장소 최상위에 배치. CSV·이미지 연결, 검증, 이미지 전처리, Dataset·DataLoader 생성 |
| `cxr_preprocessing_uignore.py` | 저장소 최상위에 배치. 기본 전처리를 재사용하고 불확실 라벨을 `-100`으로 변환 |
| `~/eva_x_project/eva_x.py` | 공식 EVA-X 모델 정의. 현재 학습 코드가 참조하는 외부 경로 |
| `~/eva_x_project/weights/eva_x_small_patch16_merged520k_mim.pt` | 공개 사전학습 가중치. 현재 코드가 참조하는 외부 경로 |

전처리 파일 두 개는 모두 필요하지만 **각각 실행할 필요는 없습니다.** 학습 파일 실행 시 자동으로 불러옵니다. 이미지를 두 번 전처리하는 구조가 아니라, 기본 이미지 전처리에 U-ignore 라벨 처리를 추가한 구조입니다.

학습 코드 자체에는 기존 DenseNet 학습 파일을 실행 시 import하는 의존성은 없습니다.

## 3. 데이터 구성

| 구분 | CSV | 이미지 수 | 불확실 라벨 수 |
|---|---|---:|---:|
| Train | `data/processed/train_final.csv` | 9,999 | 2,018 |
| Validation | `data/processed/val.csv` | 1,077 | 228 |
| Test | `data/processed/test.csv` | 2,284 | 562 |
| 합계 | — | 13,360 | 2,808 |

- 영상 선택: AP·PA 포함, 측면 영상 제외.
- 불확실 라벨 수는 **이미지 수가 아니라 소견별 라벨 개수**입니다.
- Test는 기존 DataLoader의 split 무결성 검사·파일 연결 확인을 위해 준비되지만, 학습 손실 계산 및 최적 모델 선택에는 사용하지 않습니다.
- 현재 전처리의 `CSV_NAMES["train"]`은 **`train_final.csv`**입니다. `train.csv`로 바꾸지 않습니다.
- 식별자·소견 라벨·촬영 방향·split 컬럼을 검증하고, split 간 환자·검사·이미지 식별자 중복 및 이미지 누락이 있으면 중단합니다.

실험 서버의 이미지 경로는 `/home/team30103/physionet_data`이며, 이번 실행은 **flat 레이아웃**(`IMAGE_ROOT/dicom_id.jpg`)을 사용했습니다. 다른 환경에서는 `IMAGE_ROOT`와 `image_layout`을 실제 구조에 맞춰 수정해야 합니다.

## 4. 이미지 전처리 및 라벨 처리

### 이미지 전처리

1. 흑백 영상으로 변환.
2. 긴 변을 기준으로 검은 패딩을 추가해 정사각형으로 구성.
3. Bilinear 보간으로 `224 × 224` 크기로 조정.
4. 픽셀값을 `0~1`로 변환하고 동일한 흑백값을 3채널로 복제.
5. EVA-X 정규화 적용.

```python
MEAN = (0.49185243, 0.49185243, 0.49185243)
STD = (0.28509309, 0.28509309, 0.28509309)
```

학습 파일이 `cxr_preprocessing` 모듈의 평균·표준편차를 지정한 뒤 DataLoader를 생성합니다. 기본 전처리 파일의 ImageNet 정규화 상수를 직접 수정한 것은 아닙니다. **별도의 평가·시각화 프로그램에서도 위 정규화를 일치시켜야 합니다.** DenseNet과 EVA-X를 한 프로세스에서 함께 다룰 때는 모듈 전역 정규화 값이 공유된다는 점도 고려해야 합니다.

비율 유지·패딩 방식은 기존 프로젝트의 전처리를 유지한 설정이며, EVA-X 논문의 전체 학습·전처리 절차를 그대로 재현한 실험은 아닙니다. 이번 실험에서는 별도 무작위 데이터 증강을 추가하지 않았습니다.

### U-ignore

| 원본 CSV 라벨 | 중간 변환 | 최종 학습 라벨 | 의미 |
|---:|---:|---:|---|
| 0 | 0 | 0 | 음성 |
| 1 | 1 | 1 | 양성 |
| -1 | 2 | -100 | 해당 소견의 손실·평가 계산에서 무시 |

- 불확실 라벨이 있는 **이미지 전체를 제외하지 않습니다.** 같은 이미지의 확정된 다른 소견은 학습에 사용합니다.
- 불확실을 음성으로 바꾸는 방식도 아닙니다.
- NaN은 자동으로 음성 처리하지 않고 기존 전처리에서 오류로 처리합니다.

## 5. 학습 설정

| 항목 | 설정 |
|---|---|
| 분류기 | EVA-X의 `head`를 `Linear(in_features, 6)`으로 교체 |
| 학습 범위 | 분류기를 포함한 모델 전체 |
| 손실함수 | 소견별 2-class Cross Entropy |
| 손실 집계 | 유효한 소견 라벨의 손실 합 / 유효 라벨 수 |
| 모든 라벨이 무시 대상인 배치 | 역전파·업데이트 생략 |
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Weight decay | `1e-4` |
| Batch size | 4 |
| 최대 epoch | 10 |
| Early stopping patience | 3 |
| 최적 체크포인트 선택 | 최소 validation loss |
| Seed | 42 |
| Input size | `3 × 224 × 224` |
| DataLoader workers | 0 |

현재 학습 파일은 기존 기본 U-ignore 학습 루프를 재사용하며, 별도 학습률 스케줄러·클래스 가중치·AMP는 추가하지 않았습니다.

## 6. 학습 환경 및 사전 준비

실제 작동을 확인한 환경입니다. 아래 조합이 모든 환경의 유일한 요구사항이라는 의미는 아닙니다.

| 항목 | 확인한 환경 |
|---|---|
| Python | 3.12.15 |
| PyTorch | 2.10.0+cu128 |
| CUDA — PyTorch 빌드 | 12.8 |
| GPU | NVIDIA GeForce RTX 5090 |
| timm | 0.9.16 |

기존 PyTorch 환경에서 추가한 패키지:

```bash
python -m pip install "timm==0.9.16"
```

전처리에는 기존 프로젝트의 `torchvision`, `numpy`, `pandas`, `Pillow`, `matplotlib` 환경도 필요합니다.

모델 코드와 공개 가중치는 다음 경로에 준비했습니다.

```bash
mkdir -p ~/eva_x_project/weights

curl -fL --retry 3 -o ~/eva_x_project/eva_x.py \
  https://raw.githubusercontent.com/hustvl/EVA-X/main/eva_x.py

curl -fL --retry 3 \
  -o ~/eva_x_project/weights/eva_x_small_patch16_merged520k_mim.pt \
  https://huggingface.co/MapleF/eva_x/resolve/main/eva_x_small_patch16_merged520k_mim.pt
```

### PyTorch 로딩 호환성 수정

PyTorch 2.6 이후 `torch.load()`의 `weights_only` 기본값 변경으로, 공식 체크포인트의 NumPy 값에서 `Weights only load failed` 오류가 발생했습니다. 이번 실험에서는 **공식 저장소가 안내한 출처에서 받은 해당 체크포인트에 한해** `eva_x.py`의 로딩 코드를 다음과 같이 수정했습니다.

```python
torch.load(pretrained, map_location="cpu", weights_only=False)
```

초기 가중치 로딩 시 다음 메시지가 나왔고, 분류기는 우리 과제에 맞게 새로 생성했습니다.

```text
_IncompatibleKeys(missing_keys=['head.weight', 'head.bias'], unexpected_keys=[])
```

이 실험에서는 `head`의 두 키만 누락되는 결과를 확인했습니다. 다른 키가 추가로 누락되거나 예상 밖 키가 나타나면 별도로 확인해야 합니다.

## 7. GitHub 하위 폴더 이동 시 경로 수정

**실험 당시 학습 파일은 저장소 최상위에 있었습니다.** 보유한 원본 파일을 `modle/full_training/EVA_X_S/`로 이동만 하면, 기존의 `PROJECT_DIR = Path(__file__).resolve().parent`가 EVA-X-S 폴더를 가리키게 됩니다. 또한 최상위 전처리 모듈을 찾을 수 없을 수 있습니다.

실제로 하위 폴더에서 실행하려면 다음 두 부분을 반영해야 합니다. 이 README가 `.py` 파일을 자동으로 수정하지는 않습니다.

### ① 최상위 전처리 모듈을 import하기 전에 저장소 루트 등록

`Path`, `sys`를 import한 뒤, **`import cxr_preprocessing` 및 `from cxr_preprocessing_uignore import ...`보다 위**에 배치합니다.

```python
import sys
from pathlib import Path

# repo / modle / full_training / EVA_X_S / train_evax_s_uignore_full.py
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
```

기존 `EVA_X_DIR` 및 `eva_x` import는 유지합니다. 다른 PC에서 실행하면 `~/eva_x_project`에도 모델 정의와 가중치를 준비해야 합니다.

### ② 기존 PROJECT_DIR 할당 교체

```python
# 기존: PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = REPO_ROOT
CSV_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_ROOT = PROJECT_DIR / "runs" / "evax_s_uignore_full"
```

`PROJECT_DIR`를 기존 값으로 다시 덮어쓰는 줄이 남지 않도록 확인합니다. 이 설정으로 CSV와 결과 폴더는 저장소 루트를 기준으로 유지됩니다. `IMAGE_ROOT`는 별도로 실제 이미지 경로에 맞춥니다.

## 8. 실행 방법

앞 절의 하위 폴더 이동용 경로 수정을 반영한 뒤, 저장소 루트에서 실행합니다.

```bash
python -u modle/full_training/EVA_X_S/train_evax_s_uignore_full.py
```

로그를 저장하면서 백그라운드 실행하는 예시:

```bash
mkdir -p runs/evax_s_uignore_full
training_log="runs/evax_s_uignore_full/console_$(date +%Y%m%d_%H%M%S).log"
nohup python -u modle/full_training/EVA_X_S/train_evax_s_uignore_full.py > "$training_log" 2>&1 &
echo "학습 프로세스 번호: $!"
tail -f "$training_log"
```

`tail -f`는 학습 종료 후에도 새 로그를 기다립니다. `Ctrl+C`는 로그 보기만 종료하며 백그라운드 학습을 중단하지 않습니다. 현재 학습 루프는 epoch 단위 결과를 출력하므로, 첫 epoch 종료 전에는 진행 출력이 없을 수 있습니다.

실행 전 임의 입력 순전파와 실제 이미지 4장의 순전파·역전파·가중치 업데이트가 정상 동작하는 것을 확인했습니다.

## 9. 전체 데이터 학습 결과

| Epoch | Train loss | Validation loss | Validation Macro F1 |
|---:|---:|---:|---:|
| 1 | 0.5349 | 0.5255 | 0.6854 |
| **2** | **0.4864** | **0.5219** | **0.7062** |
| 3 | 0.4292 | 0.5447 | 0.6786 |
| 4 | 0.3436 | 0.5884 | 0.6996 |
| 5 | 0.2447 | 0.6730 | 0.6959 |

- **최적 체크포인트: 2 epoch**, 최저 검증 손실 `0.5219` 기준.
- 3~5 epoch에서 학습 손실은 감소했지만 검증 손실은 증가하는 과적합 양상이 나타났습니다.
- 검증 손실이 3회 연속 개선되지 않아 **5 epoch에서 조기 종료**됐습니다.
- 종료 시점의 5 epoch 가중치가 아니라 저장된 **2 epoch 최적 모델**을 기준으로 아래 결과를 확인했습니다.

### 최적 모델의 검증 성능

| 소견 | 양성 F1 | 양성 Recall |
|---|---:|---:|
| Cardiomegaly | 0.5077 | 0.4502 |
| Edema | 0.5840 | 0.5530 |
| Pleural Effusion | 0.7755 | 0.7762 |
| **양성 평균 F1** | **0.6224** | — |

전체 검증 Macro F1: **0.7062**.

지표 계산 시 소견별 불확실 라벨을 제외합니다.

- **전체 Macro F1:** 소견 3개 × 음성·양성 클래스 2개의 F1, 총 6개 값의 평균.
- **양성 평균 F1:** 심비대·폐부종·흉수의 양성 F1 3개 값의 평균.
- **양성 Recall:** 해당 소견의 알려진 양성 영상 중 양성으로 예측한 비율.
- Macro F1 `0.7062`를 정확도 `70.62%`로 해석하지 않습니다.
- 이 실험의 기본 학습 로그·저장 지표에는 AUROC가 포함되지 않습니다. AUROC는 별도 추론 평가가 필요합니다.

## 10. DenseNet121과의 검증 결과 비교

아래는 전체 데이터 U-ignore 실험 간 비교이며 pilot 결과와의 비교가 아닙니다.

| 검증 지표 | DenseNet121 U-ignore | EVA-X-S U-ignore | EVA-X-S − DenseNet |
|---|---:|---:|---:|
| Cardiomegaly 양성 F1 | 0.4933 | **0.5077** | +0.0144 |
| Edema 양성 F1 | 0.5642 | **0.5840** | +0.0198 |
| Pleural Effusion 양성 F1 | **0.7896** | 0.7755 | −0.0141 |
| **양성 평균 F1** | 0.6157 | **0.6224** | **+0.0067** |
| 전체 Macro F1 | **0.7067** | 0.7062 | −0.0005 |

### 해석 및 한계

- EVA-X-S는 심비대·폐부종 양성 F1이 소폭 높았지만 흉수에서는 낮았습니다.
- 전체 Macro F1은 거의 같았으며, 양성 평균 F1 차이도 작아 뚜렷한 성능 개선으로 판단하지 않았습니다.
- 단일 실행 결과로 통계적 우월성을 입증한 것은 아닙니다.
- **모델 구조, 사전학습 출처, 정규화가 함께 변경됐으므로 모델 구조 하나의 효과로 해석할 수 없습니다.**
- 사전학습 모델을 사용해도 미세조정 중 과적합이 나타날 수 있음을 확인했습니다.
- 본 문서에는 시험셋 최종 성능, EVA-X Grad-CAM 결과, 자기지도 추가 사전학습 결과를 포함하지 않습니다.

## 11. 저장 산출물

각 실행은 저장소 루트 기준 `runs/evax_s_uignore_full/train_<실행시각>/`에 결과를 저장합니다.

| 파일 | 내용 |
|---|---|
| `best_model.pth` | 최적 모델·optimizer 상태, epoch, 검증 지표, 모델·전처리·학습 설정 |
| `best_val_metrics.json` | 최적 모델의 검증 손실, F1, 혼동행렬 등 |
| `history.csv` | Epoch별 학습·검증 손실 및 Macro F1 등 |
| `config.json` | 모델명, 라벨 처리, 정규화, CSV명, 학습 설정 |

이번 실험의 최적 모델 상대 경로:

```text
runs/evax_s_uignore_full/train_20261020_152249_553014/best_model.pth
```

비교한 DenseNet 실행 폴더:

```text
runs/uignore_full/train_20261019_112210_839827/
```

GitHub에는 학습 코드·README와 재현에 필요한 설정을 정리합니다. 공개 사전학습 가중치는 아래 원본 링크로 안내하고, 대용량 `.pt`·`.pth` 및 원본 CXR 데이터는 일반 코드 업로드 대상에서 제외합니다. 학습 데이터는 MIMIC-CXR 이용 권한을 갖춘 환경에서 별도로 준비해야 합니다.

## 12. 참고 자료

- [EVA-X 공식 저장소](https://github.com/hustvl/EVA-X)
- [공식 모델 정의 eva_x.py](https://github.com/hustvl/EVA-X/blob/main/eva_x.py)
- [EVA-X-S 공개 사전학습 가중치](https://huggingface.co/MapleF/eva_x/blob/main/eva_x_small_patch16_merged520k_mim.pt)
- [EVA-X 공식 분류 전처리 코드](https://github.com/hustvl/EVA-X/blob/main/classification/utils/datasets.py)
- [PyTorch 2.10 체크포인트 직렬화 문서](https://docs.pytorch.org/docs/2.10/notes/serialization.html)

외부 모델 코드와 가중치의 출처·라이선스는 해당 공식 저장소의 안내를 따릅니다.
