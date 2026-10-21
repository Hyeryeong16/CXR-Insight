# GitHub 업로드 순서

## 업로드할 위치

브랜치: `feature/model`

ZIP 안의 `full_training/densenet121_umulticlass/` 폴더 구조를 로컬 저장소 최상위에 그대로 복사합니다. 이 폴더에는 원본 코드 3개와 README, 업로드 안내가 들어 있습니다. 기존 `pilot_training/pilot_resnet/` 파일을 덮어쓰지 않습니다.

| 순서 | 파일 | 커밋 메시지 예시 |
|---|---|---|
| 1 | `train_full.py` | Add DenseNet121 U-multiclass full training baseline |
| 2 | `train_full_weighted.py` | Add class-weighted U-multiclass full training |
| 3 | `train_full_weighted_scheduler.py` | Add scheduler and validation AUROC experiment |
| 4 | `README.md`, `UPLOAD_ORDER.md` | Document U-multiclass full training experiments |

먼저 이 U-multiclass 묶음을 올리고, 다음에 앞서 준비한 U-ignore 묶음을 올리면 실험 흐름을 설명하기 쉽습니다. 이 순서는 정리용 업로드 순서이며 다른 실험들과의 정확한 실행 시간 순서를 뜻하지 않습니다.

## GitHub 웹에서 업로드

1. 저장소에서 `feature/model` 브랜치를 선택합니다.
2. 저장소 최상위의 **Add file → Upload files** 화면에서 ZIP을 푼 `full_training` 폴더를 드래그해 경로를 확인하고 커밋합니다. ZIP 파일 자체를 올리지 않습니다.
3. 위 표의 순서대로 커밋하려면 `full_training/densenet121_umulticlass/` 폴더 안에 파일을 차례로 추가합니다.

## 터미널에서 업로드

복사 후 로컬 저장소 최상위에서 먼저 상태와 브랜치를 확인합니다.

```bash
git status -sb
```

현재 브랜치가 `feature/model`인 상태에서 다음 명령을 한 줄씩 실행합니다. 다른 브랜치라면 미커밋 변경을 확인한 뒤 기존 작업 절차에 따라 전환합니다.

```bash
git add full_training/densenet121_umulticlass/train_full.py
git commit -m "Add DenseNet121 U-multiclass full training baseline"

git add full_training/densenet121_umulticlass/train_full_weighted.py
git commit -m "Add class-weighted U-multiclass full training"

git add full_training/densenet121_umulticlass/train_full_weighted_scheduler.py
git commit -m "Add scheduler and validation AUROC experiment"

git add full_training/densenet121_umulticlass/README.md full_training/densenet121_umulticlass/UPLOAD_ORDER.md
git commit -m "Document U-multiclass full training experiments"

git push origin feature/model
```

## 포함 범위

이번 묶음은 사용자 첨부 원본 3개를 확인한 결과입니다. 코드 존재와 코드 차이를 확인했으며, 실제 성능 비교에는 실행 폴더의 `config.json`, `history.csv`, `best_val_metrics.json`이 필요합니다.

같은 서버 목록에 있던 `train_full_augmented.py`, `train_full_wd.py`, `train_full_resnet50.py`, `train_full_resnet50_uignore.py`와 EVA-X는 이번 묶음에 포함하지 않았습니다. 별도 원본 확인 후 정리할 항목입니다.
