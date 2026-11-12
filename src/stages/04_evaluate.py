import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    log_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
import mlflow

from src.config.mlflow_helper import load_params, get_or_create_shared_run
from src.utils.vision_arch import safe_load_pickle
from src.utils.logger import get_logger


def calculate_all_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    clipped_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall_sensitivity": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "specificity": round(float(specificity), 4),
        "npv": round(float(npv), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "log_loss": round(float(log_loss(y_true, clipped_prob, labels=[0, 1])), 4),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def save_evaluation_plots(y_test: np.ndarray, prob_dict: dict, thresh_dict: dict, plots_dir: Path):
    roc_path = plots_dir / "multimodal_roc_comparison.png"
    pr_path = plots_dir / "multimodal_pr_comparison.png"
    cm_path = plots_dir / "multimodal_confusion_matrices.png"

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, probs in prob_dict.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_val = roc_auc_score(y_test, probs)
        ax.plot(fpr, tpr, linewidth=2.5, label=f"{name} (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--", label="Chance (0.5000)")
    ax.set_title("Unimodal vs. Multimodal ROC Curve")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, probs in prob_dict.items():
        prec, rec, _ = precision_recall_curve(y_test, probs)
        pr_val = average_precision_score(y_test, probs)
        ax.plot(rec, prec, linewidth=2.5, label=f"{name} (PR-AUC = {pr_val:.4f})")
    ax.set_title("Unimodal vs. Multimodal Precision-Recall Curve")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(pr_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(prob_dict), figsize=(5 * len(prob_dict), 4.5))
    for ax, (name, probs) in zip(axes, prob_dict.items()):
        t = thresh_dict[name]
        cm = confusion_matrix(y_test, (probs >= t).astype(int), labels=[0, 1])
        f1_v = f1_score(y_test, (probs >= t).astype(int), zero_division=0)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, xticklabels=["Healthy", "PD"], yticklabels=["Healthy", "PD"])
        ax.set_title(f"{name}\n(F1 = {f1_v:.4f})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    fig.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return [roc_path, pr_path, cm_path]


def main():
    logger= get_logger("evaluate_multimodal")
    params = load_params()
    min_f1 = params["train"]["min_f1_to_save"]

    proc_dir = Path("data/processed")
    models_dir = Path("artifacts/models")
    plots_dir = Path("artifacts/plots")
    metrics_dir = Path("artifacts/metrics")
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    paired_splits = safe_load_pickle(proc_dir / "paired_multimodal_splits.pkl")
    candidates = safe_load_pickle(models_dir / "all_multimodal_candidates.pkl")
    best_bundle_path = models_dir / "multimodal_best_with_schema.pkl"
    best_bundle = safe_load_pickle(best_bundle_path)

    mm_test = paired_splits["testing"]
    y_test = mm_test["labels"]
    thresholds = best_bundle["schema"]["thresholds"]

    img_metrics = calculate_all_metrics(y_test, mm_test["img_prob"], threshold=thresholds["image_only_fallback"])
    voc_metrics = calculate_all_metrics(y_test, mm_test["voc_prob"], threshold=thresholds["voice_only_fallback"])

    leaderboard_rows = [
        {"model_tag": "unimodal_spiral_image", "fusion_type": "unimodal", "classifier": "mobilenet_v3_large", "threshold": thresholds["image_only_fallback"], **img_metrics},
        {"model_tag": "unimodal_voice", "fusion_type": "unimodal", "classifier": "svm_et_ensemble", "threshold": thresholds["voice_only_fallback"], **voc_metrics},
    ]

    for tag, info in candidates.items():
        m = calculate_all_metrics(y_test, info["test_probs"], threshold=info["threshold"])
        leaderboard_rows.append({
            "model_tag": tag,
            "fusion_type": info["fusion_type"],
            "classifier": info["classifier_name"],
            "threshold": info["threshold"],
            **m,
        })

    leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values(
        by=["f1_score", "roc_auc", "mcc"], ascending=False
    ).reset_index(drop=True)

    mm_only_df = leaderboard_df[leaderboard_df["fusion_type"] != "unimodal"].reset_index(drop=True)
    best_mm_row = mm_only_df.iloc[0].to_dict()
    best_mm_tag = best_mm_row["model_tag"]
    best_mm_probs = candidates[best_mm_tag]["test_probs"]

    prob_dict = {
        "Spiral Image Only": mm_test["img_prob"],
        "Voice Only": mm_test["voc_prob"],
        f"Multimodal ({best_mm_tag})": best_mm_probs,
    }
    thresh_dict = {
        "Spiral Image Only": thresholds["image_only_fallback"],
        "Voice Only": thresholds["voice_only_fallback"],
        f"Multimodal ({best_mm_tag})": best_mm_row["threshold"],
    }

    plot_files = save_evaluation_plots(y_test, prob_dict, thresh_dict, plots_dir)

    leaderboard_csv = metrics_dir / "multimodal_leaderboard.csv"
    leaderboard_df.to_csv(leaderboard_csv, index=False)

    summary_json = metrics_dir / "evaluation_summary.json"
    summary_payload = {
        "unimodal_image_test": img_metrics,
        "unimodal_voice_test": voc_metrics,
        "best_multimodal_tag": best_mm_tag,
        "best_multimodal_test": {k: best_mm_row[k] for k in img_metrics.keys()},
    }
    with open(summary_json, "w") as f:
        json.dump(summary_payload, f, indent=4)

    with get_or_create_shared_run(params, create_new=False):
        for k, v in img_metrics.items():
            mlflow.log_metric(f"eval_unimodal_img_{k}", v)
        for k, v in voc_metrics.items():
            mlflow.log_metric(f"eval_unimodal_voice_{k}", v)
        for k in img_metrics.keys():
            mlflow.log_metric(f"eval_multimodal_best_{k}", float(best_mm_row[k]))

        for p_file in plot_files:
            mlflow.log_artifact(str(p_file), artifact_path="plots/multimodal_evaluation")

        mlflow.log_artifact(str(leaderboard_csv), artifact_path="metrics")
        mlflow.log_artifact(str(summary_json), artifact_path="metrics")

        qualifies = float(best_mm_row["f1_score"]) > min_f1
        mlflow.set_tag("mm_model_saved", str(qualifies))
        if qualifies:
            mlflow.log_artifact(str(best_bundle_path), artifact_path="models/multimodal_best_bundle")

    
    logger.info("=== EVALUATION LEADERBOARD ===")
    logger.info(leaderboard_df[["model_tag", "fusion_type", "f1_score", "roc_auc", "pr_auc", "accuracy", "recall_sensitivity", "specificity", "mcc"]].to_string(index=False))


if __name__ == "__main__":
    main()