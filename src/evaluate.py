"""Evaluate the production FNN on the held-out test set."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from scipy.sparse import csr_matrix, hstack

from preprocessing import build_feature_frame, load_dataset, make_splits


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"


def plot_confusion_matrix(cm: np.ndarray, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Legitimate", "Phishing"])
    ax.set_yticks([0, 1], ["Legitimate", "Phishing"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test set)")
    for i in range(2):
        for j in range(2):
            colour = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color=colour)
    fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def evaluate() -> dict:
    model = joblib.load(MODELS_DIR / "fnn_model.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    vectoriser = joblib.load(MODELS_DIR / "vectorizer.joblib")
    with open(MODELS_DIR / "model_card.json", "r", encoding="utf-8") as f:
        card = json.load(f)
    # The threshold was chosen on the validation split during training.
    # Evaluating at 0.5 instead would report a model nobody is running.
    threshold = float(card.get("decision_threshold", 0.5))

    df = load_dataset()
    X, y = build_feature_frame(df)
    data = make_splits(X, y, df["host"].astype(str))

    y_test = data["y_test"].values
    F_test = hstack([
        csr_matrix(scaler.transform(data["X_test"])),
        vectoriser.transform(data["hosts_test"]),
    ]).tocsr()

    y_proba = model.predict_proba(F_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "model_version": card.get("model_version", "unknown"),
        "dataset": card.get("dataset", "unknown"),
        "decision_threshold": threshold,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "test_set_size": int(len(y_test)),
        "class_distribution_test": {
            "legitimate": int((y_test == 0).sum()),
            "phishing": int((y_test == 1).sum()),
        },
        "feature_names": data["feature_names"],
    }

    print(f"Model: {metrics['model_version']}  (threshold {threshold:.2f})")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")

    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    plot_confusion_matrix(cm, RESULTS_DIR / "confusion_matrix.png")
    print(f"Saved metrics to {RESULTS_DIR/'metrics.json'}")
    return metrics


if __name__ == "__main__":
    evaluate()
