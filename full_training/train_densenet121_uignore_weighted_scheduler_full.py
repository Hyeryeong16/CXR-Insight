"""DenseNet121 + U-ignore + Train class weights + ReduceLROnPlateau.

Place beside train_densenet121_uignore_full.py and the preprocessing modules.
Only the training objective is weighted. Validation and model selection retain
the original unweighted CE summed over all known labels / known-label count.
No Test inference is performed. Original training files are not changed.

Fresh ImageNet initialization, not a continuation of a saved trained model.
Scheduler monitors unweighted Validation CE, halves LR after two consecutive
non-improving epochs (patience=1), with minimum LR 1e-6. Early-stop patience
remains 3 and is NOT reset by LR reduction. Maximum epochs remains 10.
History records LR used for the completed epoch and LR for the next epoch.
Validation AUROC is computed per finding from positive softmax probabilities,
excluding that finding's ignored labels. Mean AUROC is the unweighted average
of the three finding AUROCs; it is undefined if any finding lacks both classes.
No AUROC is computed on Train or Test, and model selection still uses Val CE.

python train_densenet121_uignore_weighted_scheduler_full.py --baseline
python train_densenet121_uignore_weighted_scheduler_full.py --self-test
python -u train_densenet121_uignore_weighted_scheduler_full.py
python train_densenet121_uignore_weighted_scheduler_full.py --evaluate-run /path/to/completed/run

--evaluate-run re-evaluates only best_model.pth. It writes selected_val_auroc.json
and history_with_auroc.csv (AUROC only on the selected epoch's row), preserving
the original history.csv. A live process will not pick up edits to this file.
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
OUTPUT_ROOT = PROJECT_DIR / "runs" / "uignore_weighted_scheduler_full"
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
    scheduler_factor: float = 0.5
    scheduler_patience: int = 1
    scheduler_min_lr: float = 1e-6


def make_scheduler(optimizer, config):
    # threshold=0 matches the existing strict minimum-loss selection rule.
    # No deprecated verbose argument; reductions are logged explicitly.
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.scheduler_factor,
        patience=config.scheduler_patience, threshold=0.0,
        threshold_mode="abs", cooldown=0, min_lr=config.scheduler_min_lr,
    )


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


def compute_validation_auroc(targets_by_finding, probabilities_by_finding):
    from sklearn.metrics import roc_auc_score

    if len(targets_by_finding) != 3 or len(probabilities_by_finding) != 3:
        raise ValueError("Expected targets and positive probabilities for 3 findings.")
    scores = {}
    for finding, targets, probabilities in zip(
        TARGET_LABELS, targets_by_finding, probabilities_by_finding
    ):
        if len(targets) != len(probabilities):
            raise ValueError(f"{finding}: target/probability lengths differ.")
        known_targets, known_probabilities = [], []
        for target, probability in zip(targets, probabilities):
            if target == IGNORE_INDEX:
                continue
            if target not in (0, 1):
                raise ValueError(f"{finding}: unexpected target {target}.")
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"{finding}: invalid positive probability.")
            known_targets.append(target)
            known_probabilities.append(probability)
        scores[finding] = (
            float(roc_auc_score(known_targets, known_probabilities))
            if len(set(known_targets)) == 2 else None
        )
    valid_scores = [value for value in scores.values() if value is not None]
    return {
        "auroc_per_finding": scores,
        "mean_auroc": sum(valid_scores) / 3 if len(valid_scores) == 3 else None,
        "auroc_valid_findings": len(valid_scores),
    }


def auroc_history_fields(metrics):
    scores = metrics["auroc_per_finding"]
    return {
        "val_auroc_cardiomegaly": scores["Cardiomegaly"],
        "val_auroc_edema": scores["Edema"],
        "val_auroc_pleural_effusion": scores["Pleural Effusion"],
        "val_mean_auroc": metrics["mean_auroc"],
    }


def format_metric(value):
    return "N/A" if value is None else f"{value:.4f}"


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
    auc_targets = [[], [], []]
    auc_probabilities = [[], [], []]

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
            positive_probabilities = (
                logits.detach().softmax(dim=-1)[:, :, 1].cpu() if not is_train else None
            )
            for i in range(3):
                valid = targets[:, i].ne(IGNORE_INDEX)
                indices = targets[valid, i] * 2 + predictions[valid, i]
                confusion[i] += torch.bincount(indices, minlength=4).reshape(2, 2)
                if not is_train:
                    auc_targets[i].extend(targets[valid, i].tolist())
                    auc_probabilities[i].extend(positive_probabilities[valid, i].tolist())

    if total_valid == 0:
        raise RuntimeError("There are no known labels in this split.")
    metrics = {
        "loss": objective_sum / objective_denominator,
        "unweighted_loss": ce_sum / total_valid,
        "n_images": total_images,
        "valid_label_count": total_valid,
        "skipped_all_ignored_batches": skipped,
        **metrics_from_confusion(confusion),
    }
    if not is_train:
        metrics.update(compute_validation_auroc(auc_targets, auc_probabilities))
    return metrics


def print_metrics(metrics, title):
    print(f"\n{title}", flush=True)
    print(f"Images: {metrics['n_images']} | unweighted val loss: {metrics['loss']:.4f}")
    print(f"Macro F1 (2-class): {metrics['macro_f1']:.4f}")
    positive_f1 = sum(row[1] for row in metrics["f1_per_class"]) / 3
    print(f"Mean positive F1: {positive_f1:.4f}")
    if "auroc_per_finding" in metrics:
        print("Validation AUROC:", " | ".join(
            f"{finding} {format_metric(metrics['auroc_per_finding'][finding])}"
            for finding in TARGET_LABELS
        ))
        print("Validation mean AUROC:", format_metric(metrics["mean_auroc"]))
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
    self_test_scheduler()
    self_test_auroc()
    print("Self-test passed: masking, weighted denominator, backward, Train counts, scheduler, AUROC.", flush=True)


def self_test_auroc():
    # Known rankings, tied scores, masked labels, and undefined single-class AUC.
    result = compute_validation_auroc(
        [[0, 0, 1, 1], [0, IGNORE_INDEX, 1], [0, 1]],
        [[0.1, 0.4, 0.35, 0.8], [0.1, 0.99, 0.9], [0.9, 0.1]],
    )
    assert result["auroc_per_finding"] == {
        "Cardiomegaly": 0.75, "Edema": 1.0, "Pleural Effusion": 0.0,
    }
    assert math.isclose(result["mean_auroc"], 1.75 / 3)
    fields = auroc_history_fields(result)
    assert fields["val_auroc_edema"] == 1.0
    assert fields["val_mean_auroc"] == result["mean_auroc"]
    tied = compute_validation_auroc([[0, 1]] * 3, [[0.5, 0.5]] * 3)
    assert tied["mean_auroc"] == 0.5
    undefined = compute_validation_auroc(
        [[0, 0], [0, 1], [IGNORE_INDEX]], [[0.1, 0.2], [0.1, 0.9], [0.5]],
    )
    assert undefined["auroc_valid_findings"] == 1
    assert undefined["auroc_per_finding"]["Cardiomegaly"] is None
    assert undefined["auroc_per_finding"]["Pleural Effusion"] is None
    assert undefined["mean_auroc"] is None


def self_test_scheduler():
    # Replay the previous run's losses to check exact LR timing and interaction
    # with unchanged early stopping, without opening images or using a GPU.
    config = TrainConfig()
    parameter = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=config.learning_rate)
    scheduler = make_scheduler(optimizer, config)
    losses = [0.5706, 0.5402, 0.5798, 0.5255, 0.5522, 0.6048, 0.6326]
    expected_next_lrs = [1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 5e-5, 5e-5]
    best, bad_epochs, stop_epoch = math.inf, 0, None
    for epoch, (loss, expected_lr) in enumerate(zip(losses, expected_next_lrs), 1):
        scheduler.step(loss)
        assert math.isclose(optimizer.param_groups[0]["lr"], expected_lr)
        if loss < best:
            best, bad_epochs = loss, 0
        else:
            bad_epochs += 1
        if bad_epochs >= config.patience:
            stop_epoch = epoch
    assert stop_epoch == 7
    # A new best resets scheduler patience; LR changes apply to the next epoch.
    scheduler.step(0.5)
    scheduler.step(0.51)
    assert math.isclose(optimizer.param_groups[0]["lr"], 5e-5)
    scheduler.step(0.52)
    assert math.isclose(optimizer.param_groups[0]["lr"], 2.5e-5)
    for _ in range(40):
        scheduler.step(1.0)
    assert math.isclose(optimizer.param_groups[0]["lr"], config.scheduler_min_lr)
    # Check that scheduler/optimizer states can be restored together.
    other_optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1e-4)
    other_scheduler = make_scheduler(other_optimizer, config)
    other_optimizer.load_state_dict(optimizer.state_dict())
    other_scheduler.load_state_dict(scheduler.state_dict())
    scheduler.step(1.1)
    other_scheduler.step(1.1)
    assert other_scheduler.state_dict() == scheduler.state_dict()
    assert other_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]


def fit(model, train_loader, val_loader, config, device, class_weights, counts, ignored):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = make_scheduler(optimizer, config)
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
        "validation_auroc_policy": {
            "score": "positive-class softmax probability (class index 1)",
            "mask": "exclude IGNORE_INDEX independently for each finding",
            "aggregation": "one AUROC over the entire Validation split per finding; unweighted mean of 3",
            "undefined": "null in JSON / blank in CSV; mean undefined unless all 3 AUROCs exist",
            "selection": "AUROC is reporting only; minimum Val loss still selects the model",
        },
        "baseline_metrics_path": str(BASELINE_DIR / "best_val_metrics.json"),
        "scheduler": {
            "name": "ReduceLROnPlateau", "monitor": "unweighted val loss",
            "mode": "min", "factor": config.scheduler_factor,
            "patience": config.scheduler_patience, "threshold": 0.0,
            "threshold_mode": "abs", "cooldown": 0,
            "min_lr": config.scheduler_min_lr,
            "step_timing": "after Validation; reduced LR applies to next epoch",
        },
        "early_stopping_policy": "3 consecutive non-improving epochs; not reset by LR reduction",
        "initialization": "fresh ImageNet pretrained; no resume from trained checkpoint",
        "comparison_reference": {
            "model": "DenseNet121 U-ignore weighted, no scheduler",
            "val_loss": 0.5255, "macro_f1": 0.7118, "mean_positive_f1": 0.6390,
        },
    }
    (run_dir / "config.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    history, best_loss, bad_epochs = [], math.inf, 0
    print("Output directory:", run_dir, flush=True)
    print(f"Scheduler: ReduceLROnPlateau | factor={config.scheduler_factor} | "
          f"patience={config.scheduler_patience} (reduce after 2 non-improving epochs) | "
          f"min_lr={config.scheduler_min_lr:g} | early-stop patience={config.patience}", flush=True)
    for epoch in range(1, config.epochs + 1):
        lr_used = float(optimizer.param_groups[0]["lr"])
        train = run_epoch(model, train_loader, device, optimizer, class_weights)
        val = run_epoch(model, val_loader, device)
        if not math.isfinite(val["loss"]):
            raise RuntimeError("Validation loss is NaN/Inf.")
        scheduler.step(val["loss"])
        lr_next = float(optimizer.param_groups[0]["lr"])
        lr_reduced = lr_next < lr_used
        history.append({
            "epoch": epoch,
            "lr_used": lr_used, "lr_next": lr_next, "lr_reduced": lr_reduced,
            "train_weighted_loss": train["loss"], "train_unweighted_ce": train["unweighted_loss"],
            "val_loss": val["loss"], "train_macro_f1": train["macro_f1"],
            "val_macro_f1": val["macro_f1"], "val_mean_positive_f1": val["mean_positive_f1"],
            **auroc_history_fields(val),
            "train_valid_labels": train["valid_label_count"], "val_valid_labels": val["valid_label_count"],
        })
        with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        print(f"Epoch {epoch}/{config.epochs} | train weighted loss {train['loss']:.4f} | "
              f"train CE {train['unweighted_loss']:.4f} | val loss {val['loss']:.4f} | "
              f"val macro F1(2-class) {val['macro_f1']:.4f} | "
              f"val mean positive F1 {val['mean_positive_f1']:.4f} | "
              f"lr used {lr_used:.2e} | lr next {lr_next:.2e}", flush=True)
        print("Val AUROC — Cardiomegaly/Edema/Pleural Effusion:",
              [format_metric(val["auroc_per_finding"][finding]) for finding in TARGET_LABELS],
              "| mean", format_metric(val["mean_auroc"]), flush=True)
        if lr_reduced:
            print(f"Learning rate reduced: {lr_used:.2e} -> {lr_next:.2e}; "
                  "applies to next epoch if training continues.", flush=True)
        if val["loss"] < best_loss:
            best_loss, bad_epochs = val["loss"], 0
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "lr_used": lr_used, "lr_next": lr_next,
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
    print_metrics(checkpoint["val_metrics"], "Weighted + scheduler model: selected Validation result")
    reference = metadata["comparison_reference"]
    print("Compared with weighted model WITHOUT scheduler (rounded stored reference):")
    for key in ("loss", "macro_f1", "mean_positive_f1"):
        reference_key = "val_loss" if key == "loss" else key
        print(f"{key}: {checkpoint['val_metrics'][key]:.4f} "
              f"(reference {reference[reference_key]:.4f})")
    return best_path


def write_selected_history(run_dir, selected_epoch, metrics):
    """Create a derived CSV; never invent AUROCs for unsaved epochs."""
    source = run_dir / "history.csv"
    with source.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields, rows = list(reader.fieldnames or []), list(reader)
    if "epoch" not in fields:
        raise ValueError("history.csv has no epoch column.")
    matching = [row for row in rows if int(row["epoch"]) == selected_epoch]
    if len(matching) != 1:
        raise ValueError("Expected exactly one history row for the selected epoch.")
    extra = auroc_history_fields(metrics)
    for key in extra:
        if key not in fields:
            fields.append(key)
    matching[0].update(extra)
    output = run_dir / "history_with_auroc.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output


def evaluate_completed_run(run_directory):
    """Re-evaluate best_model.pth after a run has finished; no optimizer steps."""
    from cxr_preprocessing_uignore import (
        IGNORE_INDEX as preprocessing_ignore,
        TARGET_LABELS as preprocessing_targets,
        build_dataloaders,
    )

    if preprocessing_ignore != IGNORE_INDEX or tuple(preprocessing_targets) != TARGET_LABELS:
        raise ValueError("Preprocessing ignore value or target order differs.")
    run_dir = Path(run_directory).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    checkpoint = torch.load(run_dir / "best_model.pth", map_location="cpu", weights_only=True)
    if (checkpoint.get("architecture") != "densenet121"
            or checkpoint.get("label_strategy") != "u_ignore"
            or tuple(checkpoint.get("target_labels", [])) != TARGET_LABELS):
        raise ValueError("Expected a DenseNet121 U-ignore checkpoint with matching label order.")
    self_test()
    config = TrainConfig(**{
        key: value for key, value in checkpoint["config"].items()
        if key in TrainConfig.__dataclass_fields__
    })
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Evaluation device:", device, flush=True)
    loaders = build_dataloaders(
        csv_dir=CSV_DIR, image_root=IMAGE_ROOT, image_layout="flat",
        image_size=config.image_size, batch_size=config.batch_size,
        num_workers=config.num_workers, seed=config.seed,
    )
    model = DenseNetUIgnore(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = run_epoch(model, loaders["validate"], device)
    original = checkpoint["val_metrics"]
    # Refuse to attach AUROC if current data/preprocessing no longer reproduces
    # the checkpoint's saved Validation result.
    if (metrics["n_images"] != original["n_images"]
            or metrics["valid_label_count"] != original["valid_label_count"]
            or metrics["confusion_matrix"] != original["confusion_matrix"]
            or not math.isclose(metrics["loss"], original["loss"], rel_tol=1e-4, abs_tol=1e-5)):
        raise ValueError("Validation differs from the saved result. Check data/preprocessing before merging AUROC.")
    history_path = write_selected_history(run_dir, int(checkpoint["epoch"]), metrics)
    output = run_dir / "selected_val_auroc.json"
    output.write_text(json.dumps({
        "checkpoint": str(run_dir / "best_model.pth"),
        "selected_epoch": int(checkpoint["epoch"]),
        "scope": "Only the saved selected epoch was evaluated; unsaved epochs are not reconstructed.",
        "val_metrics": metrics,
    }, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print_metrics(metrics, f"Selected Validation result — epoch {checkpoint['epoch']}")
    print("Saved:", output)
    print("Saved:", history_path)
    print("Original history.csv preserved. AUROC is filled only for the selected epoch.")


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
    parser = argparse.ArgumentParser(description="DenseNet121 + U-ignore weighted + LR scheduler experiment")
    parser.add_argument("--baseline", action="store_true", help="Print stored baseline Validation metrics only")
    parser.add_argument("--self-test", action="store_true", help="Run small CPU loss tests without images")
    parser.add_argument("--evaluate-run", metavar="RUN_DIR", help="After training finishes, evaluate best_model.pth and write history_with_auroc.csv")
    args = parser.parse_args()
    if sum([args.baseline, args.self_test, args.evaluate_run is not None]) > 1:
        parser.error("Choose only one of --baseline, --self-test, --evaluate-run.")
    if args.baseline:
        show_baseline(required=True)
    elif args.self_test:
        self_test()
    elif args.evaluate_run:
        evaluate_completed_run(args.evaluate_run)
    else:
        main()
