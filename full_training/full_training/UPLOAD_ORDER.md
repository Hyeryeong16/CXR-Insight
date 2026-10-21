# GitHub 업로드 순서

2026-10-21 사용자 확인을 반영한 목록입니다. 원격 저장소를 직접 조회한 결과는 아닙니다.

## 이미 업로드된 항목

기존 DenseNet121 파일럿 코드에 더해 `feature/model` 브랜치의 `pilot_training/pilot_resnet/`에 다음 항목이 올라간 것으로 확인했습니다.

- `FULL_TRAINING_GUIDE.md`
- `train_pilot_resnet50_uignore.py`
- `train_pilot_resnet50_umulticlass.py`
- `train_pilot_resnet50_umulticlass_weighted.py`

이번 묶음에서는 위 파일을 다시 업로드하지 않습니다. 전체 학습 가이드가 있다는 사실만으로 실제 전체 학습 코드까지 업로드되었다고 판단하지 않았습니다.

## 이번 업로드

ZIP을 풀면 Python 파일 4개와 문서 폴더 `full_training/`가 나옵니다. 이 구조를 로컬 저장소 최상위에 복사합니다. 기존 프로젝트의 `cxr_preprocessing.py`는 그대로 사용합니다.

| 순서 | 추가할 파일 | 목적 |
|---|---|---|
| 1 | `cxr_preprocessing_uignore.py`, `train_densenet121_uignore_full.py` | 기본 전체 데이터 U-ignore 학습 기록 |
| 2 | `train_densenet121_uignore_weighted_full.py` | 클래스 불균형 대응 실험 |
| 3 | `train_densenet121_uignore_weighted_scheduler_full.py` | 학습률 조절 실험 |
| 4 | `full_training/README.md`, `full_training/UPLOAD_ORDER.md` | 실행 방법 및 업로드 이력 |

각 항목은 업로드용으로 준비한 보관본입니다. GPU 서버에서 실제 사용한 파일에 추가 수정이 있다면 그 서버 버전을 우선합니다. 실험이 진행된 정확한 시간 순서는 실행 폴더의 `config.json`과 로그로 확인할 수 있습니다.

## GitHub 웹에서 올리기

1. 저장소에서 `feature/model` 브랜치를 선택합니다.
2. 저장소 최상위에서 **Add file → Upload files**를 선택합니다.
3. 위 순서대로 파일을 올리고 각각 커밋합니다. `.py` 파일은 저장소 최상위에 배치합니다.
4. 문서는 `full_training/` 폴더 안에 올립니다. 폴더를 드래그하거나 Create new file에서 경로를 입력해 만들 수 있습니다.

## 터미널에서 올리기

파일을 로컬 저장소에 복사한 후, 저장소 최상위에서 실행합니다. Windows PowerShell에서도 한 줄씩 그대로 실행할 수 있습니다.

```bash
git status -sb
```

현재 브랜치가 `feature/model`인지 확인합니다. 다른 브랜치라면 기존 미커밋 변경을 확인한 뒤 `git switch feature/model`로 전환합니다.

```bash
git add cxr_preprocessing_uignore.py train_densenet121_uignore_full.py
git commit -m "Add DenseNet121 U-ignore full training baseline"

git add train_densenet121_uignore_weighted_full.py
git commit -m "Add class-weighted DenseNet121 full training"

git add train_densenet121_uignore_weighted_scheduler_full.py
git commit -m "Add learning rate scheduler experiment"

git add full_training/README.md full_training/UPLOAD_ORDER.md
git commit -m "Document full training experiments and upload order"

git push origin feature/model
```

기존 코드가 이미 같은 경로에 있다면 새로 덮어쓰기 전에 내용을 비교합니다. 데이터, CSV 및 모델 가중치는 이 코드 묶음에 포함되지 않습니다.

## 다음으로 원본 확인이 필요한 항목

| 항목 | 현재 상태 |
|---|---|
| DenseNet121 U-multiclass 전체 학습 | 학습 기록은 있으나 이번에 확인한 보관본에서 원본 코드 미확인 |
| ResNet50 전체 데이터 학습 | 실제 실행 여부와 원본 파일 확인 필요 |
| 전체 데이터 증강·Weight decay 비교 | 실제 사용 파일과 각 설정 확인 필요 |
| EVA-X-S U-ignore 전체 학습 | 학습 기록은 있으나 원본 코드 미확인 |
| Early stopping 해제·50 epoch | 논의한 계획이며 이번 묶음의 코드는 해당 실험이 아님 |

원본을 확보하기 전에는 실험 코드를 재구성해 실제 학습에 사용한 원본으로 표시하지 않습니다.
