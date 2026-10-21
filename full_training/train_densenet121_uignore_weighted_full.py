"""DenseNet121 + U-ignore, with weights computed from the actual Train labels.

Place beside train_densenet121_uignore_full.py and the preprocessing modules.
Only the training objective is weighted. Validation and model selection retain
the original unweighted CE summed over all known labels / known-label count.
No Test inference is performed. Original training files are not changed.

python train_densenet121_uignore_weighted_full.py --baseline
python train_densenet121_uignore_weighted_full.py --self-test
python -u train_densenet121_uignore_weighted_full.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

PROJECT_DIR = Path(__file__).resolve().parent
CSV_DIR = PROJECT_DIR / "data" / "processed"
IMAGE_ROOT = Path("/home/team30103/physionet_data")
OUTPUT_ROOT = PROJECT_DIR / "runs" / "uignore_weighted_full"
BASELINE_DIR = PROJECT_DIR / "runs" / "uignore_full" / "train_20261019_112210_839827"
TARGET_LABELS = ("Cardiomegaly", "Edema", "Pleural Effusion")
CLASS_NAMES = ("negative", "positive")
IGNORE_INDEX = -100


@dataclass
class TrainConfig:
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 3
    batch_size: int = 4
    seed: int = 42
    image_size: int = 224
    num_workers: int = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class DenseNetUIgnore(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import DenseNet121_Weights, densenet121

        self.backbone = densenet121(
            weights=DenseNet121_Weights.DEFAULT if pretrained else None
        )
        self.backbone.classifier = nn.Linear(
            self.backbone.classifier.in_features, 3 * 2
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images).reshape(-1, 3, 2)


def weights_from_train_labels(labels: torch.Tensor):
    """Use post-preprocessing labels; exclude -100 before computing weights.

    weights[f,c] = (1/sqrt(count[f,c])) / mean_c(1/sqrt(count[f,c])).
    Normalizing each pair makes the mean of the two class weights one.
    """
    labels = labels.detach().cpu()
    if labels.ndim != 2 or labels.shape[1] != 3 or labels.dtype != torch.long:
        raise ValueError("Train labels must be a long tensor with shape [N, 3].")
    if not torch.isin(labels, torch.tensor([0, 1, IGNORE_INDEX])).all().item():
        raise ValueError("Train labels must contain only 0, 1 and -100.")
    counts = torch.stack([(labels == c).sum(dim=0) for c in range(2)], dim=1)
    ignored = (labels == IGNORE_INDEX).sum(dim=0)
    if (counts == 0).any().item():
        raise ValueError(f"A Train class has no examples: {counts.tolist()}")
    weights = counts.double().rsqrt()
    weights = (weights / weights.mean(dim=1, keepdim=True)).float()
    return counts, ignored, weights


def masked_loss_parts(logits, labels, class_weights=None):
    """Return objective sum, objective denominator, unweighted CE sum, count.

    Train: weighted CE / sum of weights of valid labels.
    Validation: unweighted CE / count of valid labels, exactly as the baseline.
    Ignored labels contribute neither loss nor denominator in either case.
    """
    if logits.shape != (*labels.shape, 2):
        raise ValueError("Expected logits [B, 3, 2] and labels [B, 3].")
    losses = F.cross_entropy(
        logits.reshape(-1, 2), labels.reshape(-1),
        ignore_index=IGNORE_INDEX, reduction="none",
    ).reshape_as(labels)
    valid = labels.ne(IGNORE_INDEX)
    count = int(valid.sum().item())
    raw_sum = losses[valid].sum()
    if class_weights is None:
        return raw_sum, float(count), raw_sum, count

    if class_weights.shape != (3, 2):
        raise ValueError("Expected class weights [3, 2].")
    safe_labels = labels.masked_fill(~valid, 0)
    weights = class_weights.unsqueeze(0).expand(labels.shape[0], -1, -1)
    weights = weights.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)
    weights = weights.masked_fill(~valid, 0)
    weighted_sum = (losses * weights).sum()
    return weighted_sum, float(weights.sum().item()), raw_sum, count


def metrics_from_confusion(confusion):
    tp = confusion.diagonal(dim1=1, dim2=2).float()
    support = confusion.sum(dim=2)
    predicted = confusion.sum(dim=1)
    f1 = 2 * tp / (support + predicted).clamp_min(1)
    return {
        "macro_f1": f1.mean().item(),
        "mean_positive_f1": f1[:, 1].mean().item(),
        "f1_per_class": f1.tolist(),
        "confusion_matrix": confusion.tolist(),
    }


def run_epoch(model, loader, device, optimizer=None, class_weights=None):
    from cxr_preprocessing_uignore import check_uignore_batch

    is_train = optimizer is not None
    if is_train and class_weights is None:
        raise ValueError("Training requires Train-derived class weights.")
    if not is_train and class_weights is not None:
        raise ValueError("Validation must use the baseline unweighted loss.")
    model.train(is_train)
    objective_sum, objective_denominator, ce_sum = 0.0, 0.0, 0.0
    total_valid, total_images, skipped = 0, 0, 0
    confusion = torch.zeros((3, 2, 2), dtype=torch.long)

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            check_uignore_batch(images, labels)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            # Retain the baseline forward/skip order for this comparison.
            logits = model(images)
            numerator, denominator, raw_sum, count = masked_loss_parts(
                logits, labels, class_weights
            )
            total_images += len(images)
            if count == 0:
                skipped += 1
                continue
            loss = numerator / denominator
            if not torch.isfinite(loss).item():
                raise RuntimeError("Loss is NaN/Inf.")
            if is_train:
                loss.backward()
                optimizer.step()
            objective_sum += float(numerator.detach().item())
            objective_denominator += denominator
            ce_sum += float(raw_sum.detach().item())
            total_valid += count

            predictions = logits.detach().argmax(dim=-1).cpu()
            targets = labels.detach().cpu()
            for i in range(3):
                valid = targets[:, i].ne(IGNORE_INDEX)
                indices = targets[valid, i] * 2 + predictions[valid, i]
                confusion[i] += torch.bincount(indices, minlength=4).reshape(2, 2)

    if total_valid == 0:
        raise RuntimeError("There are no known labels in this split.")
    return {
        "loss": objective_sum / objective_denominator,
        "unweighted_loss": ce_sum / total_valid,
        "n_images": total_images,
        "valid_label_count": total_valid,
        "skipped_all_ignored_batches": skipped,
        **metrics_from_confusion(confusion),
    }


def print_metrics(metrics, title):
    print(f"\n{title}", flush=True)
    print(f"Images: {metrics['n_images']} | unweighted val loss: {metrics['loss']:.4f}")
    print(f"Macro F1 (2-class): {metrics['macro_f1']:.4f}")
    positive_f1 = sum(row[1] for row in metrics["f1_per_class"]) / 3
    print(f"Mean positive F1: {positive_f1:.4f}")
    print("Finding | Class | True count | Predicted count | Precision | Recall | F1")
    for finding, matrix in zip(TARGET_LABELS, metrics["confusion_matrix"]):
        if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
            raise ValueError("Expected 2x2 confusion matrices for U-ignore.")
        for c, name in enumerate(CLASS_NAMES):
            tp = matrix[c][c]
            support = sum(matrix[c])
            predicted = sum(row[c] for row in matrix)
            precision = f"{tp / predicted:.4f}" if predicted else "N/A"
            recall = f"{tp / support:.4f}" if support else "N/A"
            f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
            print(f"{finding} | {name} | {support} | {predicted} | "
                  f"{precision} | {recall} | {f1:.4f}")


def show_baseline(required=False):
    path = BASELINE_DIR / "best_val_metrics.json"
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        print("Baseline metrics not found:", path)
        return None
    metrics = json.loads(path.read_text(encoding="utf-8"))
    if metrics["n_images"] != 1077:
        raise ValueError("Baseline Validation size is not 1,077.")
    config_path = BASELINE_DIR / "config.json"
    if config_path.is_file():
        metadata = json.loads(config_path.read_text(encoding="utf-8"))
        if (metadata.get("architecture") != "densenet121"
                or metadata.get("label_strategy") != "u_ignore"):
            raise ValueError("Baseline metadata is not DenseNet121 + U-ignore.")
    print("Baseline file:", path)
    print_metrics(metrics, "Baseline DenseNet121 + U-ignore Validation")
    return metrics


def self_test():
    # Test loss arithmetic, ignored-label masking and backward without images.
    labels = torch.tensor([[0, -100, 1], [1, 0, -100]], dtype=torch.long)
    logits = torch.tensor([
        [[1.0, -0.5], [0.2, 0.8], [-0.1, 0.4]],
        [[0.7, 0.1], [-0.3, 0.9], [0.0, 1.0]],
    ], requires_grad=True)
    n, d, raw, count = masked_loss_parts(logits, labels)
    expected = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1),
                               ignore_index=IGNORE_INDEX, reduction="sum")
    torch.testing.assert_close(n, expected)
    assert count == 4 and d == 4
    wn, wd, _, _ = masked_loss_parts(logits, labels, torch.ones(3, 2))
    torch.testing.assert_close(wn / wd, raw / count)

    weights = torch.tensor([[0.8, 1.2], [0.5, 1.5], [0.6, 1.4]])
    wn, wd, _, _ = masked_loss_parts(logits, labels, weights)
    terms, masses = [], []
    for row in range(2):
        for finding in range(3):
            target = int(labels[row, finding])
            if target != IGNORE_INDEX:
                term = F.cross_entropy(logits[row, finding].unsqueeze(0),
                                       labels[row, finding].unsqueeze(0))
                weight = weights[finding, target]
                terms.append(term * weight)
                masses.append(weight)
    torch.testing.assert_close(wn / wd, torch.stack(terms).sum() / torch.stack(masses).sum())
    (wn / wd).backward()
    assert logits.grad[labels.eq(IGNORE_INDEX)].abs().sum().item() == 0.0
    assert logits.grad[labels.ne(IGNORE_INDEX)].abs().sum().item() > 0.0

    empty = torch.zeros(2, 3, 2, requires_grad=True)
    en, ed, _, ec = masked_loss_parts(empty, torch.full_like(labels, IGNORE_INDEX), weights)
    assert ec == 0 and ed == 0 and en.item() == 0
    en.backward()
    assert empty.grad.abs().sum().item() == 0

    sample = torch.tensor([[0, 0, 1], [1, 0, 0], [-100, 1, -100]], dtype=torch.long)
    counts, ignored, cw = weights_from_train_labels(sample)
    assert counts.tolist() == [[1, 1], [2, 1], [1, 1]]
    assert ignored.tolist() == [1, 0, 1]
    assert cw[1, 1] > cw[1, 0]
    torch.testing.assert_close(cw.mean(dim=1), torch.ones(3))
    print("Self-test passed: masking, weighted denominator, backward, Train counts.", flush=True)


def fit(model, train_loader, val_loader, config, device, class_weights, counts, ignored):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    run_dir = OUTPUT_ROOT / ("train_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    run_dir.mkdir(parents=True, exist_ok=False)
    best_path = run_dir / "best_model.pth"
    metadata = {
        "architecture": "densenet121", "label_strategy": "u_ignore",
        "class_names": list(CLASS_NAMES), "class_map": {"0": 0, "1": 1, "-1": IGNORE_INDEX},
        "target_labels": list(TARGET_LABELS), "selection_metric": "minimum val loss",
        "pretrained_imagenet": True,
        "preprocessing": "cxr_preprocessing.py / cxr_preprocessing_uignore.py",
        "config": asdict(config),
        "class_weights": class_weights.detach().cpu().tolist(),
        "train_class_counts": counts.tolist(), "train_ignored_counts": ignored.tolist(),
        "class_weight_method": "inverse square root; mean class weight 1 per finding",
        "train_loss_policy": "weighted CE sum / sum of valid label weights across all findings",
        "validation_loss_policy": "unweighted CE sum / known-label count across all findings",
        "baseline_metrics_path": str(BASELINE_DIR / "best_val_metrics.json"),
    }
    (run_dir / "config.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    history, best_loss, bad_epochs = [], math.inf, 0
    for epoch in range(1, config.epochs + 1):
        train = run_epoch(model, train_loader, device, optimizer, class_weights)
        val = run_epoch(model, val_loader, device)
        history.append({
            "epoch": epoch,
            "train_weighted_loss": train["loss"], "train_unweighted_ce": train["unweighted_loss"],
            "val_loss": val["loss"], "train_macro_f1": train["macro_f1"],
            "val_macro_f1": val["macro_f1"], "val_mean_positive_f1": val["mean_positive_f1"],
            "train_valid_labels": train["valid_label_count"], "val_valid_labels": val["valid_label_count"],
        })
        with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        print(f"Epoch {epoch}/{config.epochs} | train weighted loss {train['loss']:.4f} | "
              f"train CE {train['unweighted_loss']:.4f} | val loss {val['loss']:.4f} | "
              f"val macro F1(2-class) {val['macro_f1']:.4f} | "
              f"val mean positive F1 {val['mean_positive_f1']:.4f}", flush=True)
        if val["loss"] < best_loss:
            best_loss, bad_epochs = val["loss"], 0
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch, "val_metrics": val, **metadata}, best_path)
            (run_dir / "best_val_metrics.json").write_text(json.dumps(val, indent=2), encoding="utf-8")
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                print("Early stopping", flush=True)
                break
    print("Best checkpoint:", best_path)
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Saved model restored. Best epoch:", checkpoint["epoch"])
    print_metrics(checkpoint["val_metrics"], "Weighted model: selected Validation result")
    return best_path


def main():
    from cxr_preprocessing_uignore import (
        IGNORE_INDEX as preprocessing_ignore,
        TARGET_LABELS as preprocessing_targets,
        build_dataloaders, check_uignore_batch,
    )
    if preprocessing_ignore != IGNORE_INDEX or tuple(preprocessing_targets) != TARGET_LABELS:
        raise ValueError("Preprocessing ignore value or target order differs from this script.")
    self_test()
    config = TrainConfig()
    set_seed(config.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU(cuda) is unavailable. Activate the training environment.")
    device = torch.device("cuda")
    print("Device:", device)
    show_baseline()
    loaders = build_dataloaders(
        csv_dir=CSV_DIR, image_root=IMAGE_ROOT, image_layout="flat",
        image_size=config.image_size, batch_size=config.batch_size,
        num_workers=config.num_workers, seed=config.seed,
    )
    expected_sizes = {"train": 9999, "validate": 1077, "test": 2284}
    for split, size in expected_sizes.items():
        loader = loaders[split]
        print(f"{split}: {len(loader.dataset)} images / {len(loader)} batches")
        if len(loader.dataset) != size:
            raise ValueError(f"{split}: expected {size} images.")
    for split in ("train", "validate"):
        check_uignore_batch(*next(iter(loaders[split])))

    # Read the actual mapped labels without opening any extra images or shuffling.
    counts, ignored, weights = weights_from_train_labels(loaders["train"].dataset.labels)
    print("Train class counts/weights: negative, positive; ignored excluded")
    for label, row, ig, weight in zip(TARGET_LABELS, counts, ignored, weights):
        print(label, "counts:", row.tolist(), "ignored:", int(ig),
              "weights:", [round(v, 4) for v in weight.tolist()])
    model = DenseNetUIgnore(pretrained=True).to(device)
    fit(model, loaders["train"], loaders["validate"], config, device,
        weights.to(device), counts, ignored)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DenseNet121 + U-ignore weighted Train experiment")
    parser.add_argument("--baseline", action="store_true", help="Print stored baseline Validation metrics only")
    parser.add_argument("--self-test", action="store_true", help="Run small CPU loss tests without images")
    args = parser.parse_args()
    if args.baseline:
        show_baseline(required=True)
    elif args.self_test:
        self_test()
    else:
        main()
