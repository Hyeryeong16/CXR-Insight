# Dataset

## 1. 데이터 출처
- **데이터셋:** MIMIC-CXR-JPG
- **제공:** PhysioNet
- **공식 링크:** https://physionet.org/content/mimic-cxr-jpg/

## 2. 데이터 선정 기준
- 흉부 X-ray 정면 영상(AP/PA) 사용
- 측면 영상(LATERAL) 제외
- 분석 대상 소견 3개
  - Cardiomegaly (심비대)
  - Edema (폐부종)
  - Pleural Effusion (흉수)

## 3. 데이터 구성

| 구분 | 이미지 수 |
|---|---:|
| Train | 9,999 |
| Validation | 1,077 |
| Test | 2,284 |
| **Total** | **13,360** |

## 4. 라벨 처리
- `0`: 음성(Negative)
- `1`: 양성(Positive)
- `-1`: 불확실(Uncertain)
- `NaN`: 결측(Missing)

최종 모델에서는 **U-Ignore** 방식을 적용하여 불확실하거나 결측된 라벨을 해당 소견의 손실 계산에서 제외했습니다.

## 5. 데이터 관리
- MIMIC-CXR-JPG 원본 이미지 및 접근 제한 데이터는 GitHub에 업로드하지 않습니다.
- 환자·검사 식별자가 포함된 CSV 파일도 공개 저장소에 업로드하지 않습니다.
- 데이터 전처리 코드는 `preprocessing/` 폴더에서 관리합니다.
