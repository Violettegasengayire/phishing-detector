"""Train the production phishing detector on 2024 data.

The model is a feedforward neural network (scikit-learn's
``MLPClassifier``) over two kinds of input derived from the hostname:

* 23 engineered features (length, entropy, hyphens, digits, free TLD,
  sensitive words, ...) from ``url_features``
* character n-grams of the hostname, TF-IDF weighted

The n-grams are what carry the recall. Without them the same network
reaches 0.698 recall; with them it reaches 0.841. They also pull some
legitimate sign-in hostnames upward, which is why the decision threshold
is calibrated rather than left at 0.5 - see ``choose_threshold``.

Run ``python src/build_dataset.py`` first.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier

from preprocessing import build_feature_frame, load_dataset, make_splits
from url_features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_VERSION = "3.0.0-hostname-ngram"
DATASET_NAME = "UCI PhiUSIIL (2024) phishing hosts + Tranco legitimate hosts"

VECTORISER_PARAMS = {
    "analyzer": "char_wb",
    "ngram_range": (3, 4),
    "min_df": 5,
    "max_features": 20_000,
    "sublinear_tf": True,
}

MODEL_PARAMS = {
    "hidden_layer_sizes": (128, 64),
    "activation": "relu",
    "solver": "adam",
    "learning_rate_init": 1e-3,
    "batch_size": 512,
    "max_iter": 80,
    "early_stopping": True,
    "validation_fraction": 0.15,
    "n_iter_no_change": 8,
    "alpha": 1e-4,
    "random_state": 42,
}

#: Precision floor used to pick the operating threshold. Phishing recall
#: is what protects the user, so the threshold is chosen as the most
#: sensitive one that still keeps false alarms rare enough to be worth
#: reading. A detector that cries wolf gets ignored, which costs more
#: than the warning was worth.
MIN_PRECISION = 0.95


def choose_threshold(y_true, proba, min_precision: float = MIN_PRECISION) -> float:
    """Pick the decision threshold on validation data.

    The default 0.5 is wrong for this model. Character n-grams push some
    legitimate sign-in hostnames into the 0.6-0.8 band, so at 0.5 the
    detector flags login.microsoftonline.com. Rather than discard the
    n-grams and lose the recall they provide, the threshold is moved to
    where the two classes actually separate.

    This is fitted on validation data only. Choosing it on the test set,
    or on a hand-written list of example sites, would be tuning on the
    thing being measured.
    """
    best = (0.5, -1.0)
    for t in np.arange(0.30, 0.99, 0.01):
        pred = (proba >= t).astype(int)
        precision = precision_score(y_true, pred, zero_division=0)
        if precision < min_precision:
            continue
        recall = recall_score(y_true, pred)
        if recall > best[1]:
            best = (float(t), float(recall))
    return best[0]


def train() -> dict:
    df = load_dataset()
    X, y = build_feature_frame(df)
    data = make_splits(X, y, df["host"].astype(str))
    scaler = data["scaler"]

    vectoriser = TfidfVectorizer(**VECTORISER_PARAMS)
    A_train = vectoriser.fit_transform(data["hosts_train"])

    def combine(X_part, hosts_part):
        return hstack([
            csr_matrix(scaler.transform(X_part)),
            vectoriser.transform(hosts_part),
        ]).tocsr()

    F_train = hstack([
        csr_matrix(scaler.transform(data["X_train"])), A_train
    ]).tocsr()
    F_val = combine(data["X_val"], data["hosts_val"])

    model = MLPClassifier(**MODEL_PARAMS)
    model.fit(F_train, data["y_train"])

    proba_val = model.predict_proba(F_val)[:, 1]
    threshold = choose_threshold(data["y_val"], proba_val)
    pred_val = (proba_val >= threshold).astype(int)

    print(f"Training rows:        {len(data['y_train']):,}")
    print(f"Input dimensions:     {F_train.shape[1]:,}")
    print(f"Converged in:         {model.n_iter_} epochs")
    print(f"Chosen threshold:     {threshold:.2f}")
    print(f"Validation precision: {precision_score(data['y_val'], pred_val):.4f}")
    print(f"Validation recall:    {recall_score(data['y_val'], pred_val):.4f}")
    print(f"Validation F1:        {f1_score(data['y_val'], pred_val):.4f}")

    joblib.dump(model, MODELS_DIR / "fnn_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(vectoriser, MODELS_DIR / "vectorizer.joblib")
    with open(MODELS_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(data["feature_names"], f, indent=2)
    with open(MODELS_DIR / "model_card.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_version": MODEL_VERSION,
                "dataset": DATASET_NAME,
                "dataset_rows": int(len(df)),
                "architecture": "MLPClassifier (feedforward neural network)",
                "engineered_features": len(FEATURE_NAMES),
                "ngram_features": int(len(vectoriser.vocabulary_)),
                "input_dimensions": int(F_train.shape[1]),
                "hidden_layers": list(MODEL_PARAMS["hidden_layer_sizes"]),
                "activation": MODEL_PARAMS["activation"],
                "optimizer": MODEL_PARAMS["solver"],
                "learning_rate_init": MODEL_PARAMS["learning_rate_init"],
                "batch_size": MODEL_PARAMS["batch_size"],
                "l2_alpha": MODEL_PARAMS["alpha"],
                "early_stopping": True,
                "random_state": MODEL_PARAMS["random_state"],
                "n_iter": int(model.n_iter_),
                "decision_threshold": threshold,
                "threshold_selection": (
                    f"Most sensitive threshold on the validation split that "
                    f"keeps precision >= {MIN_PRECISION}. Not tuned on test "
                    f"data or on example sites."
                ),
                "feature_names": data["feature_names"],
                "class_mapping": {"0": "Legitimate", "1": "Phishing"},
                "label_source": (
                    "is_phishing from build_dataset.py: 1 phishing "
                    "(PhiUSIIL 2024), 0 legitimate (Tranco). Class 1 is "
                    "phishing so predict_proba[:, 1] is P(phishing)."
                ),
                "scope": "hostname only; paths and query strings are ignored",
            },
            f,
            indent=2,
        )

    loss_curve = list(map(float, model.loss_curve_))
    with open(RESULTS_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(
            {"loss_curve": loss_curve, "n_iter": int(model.n_iter_)}, f, indent=2
        )
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(loss_curve)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training loss")
        ax.set_title(f"Training loss ({MODEL_VERSION})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "training_loss.png", dpi=110)
        plt.close(fig)
    except Exception as exc:
        print(f"(matplotlib unavailable: {exc})")

    print(f"Saved model to {MODELS_DIR / 'fnn_model.joblib'}")
    return {"threshold": threshold, "n_iter": int(model.n_iter_)}


if __name__ == "__main__":
    train()
