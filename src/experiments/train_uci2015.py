"""Train the production FNN on URL-derivable features.

The architecture is a feedforward neural network (scikit-learn's
MLPClassifier). The feature set is the subset the production extractor
can genuinely compute from a URL, so training and inference share the
same representation.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier

from preprocessing import load_arff, make_splits, split_features_target


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


MODEL_PARAMS = {
    "hidden_layer_sizes": (32, 16),
    "activation": "relu",
    "solver": "adam",
    "learning_rate_init": 1e-3,
    "batch_size": 64,
    "max_iter": 300,
    "early_stopping": True,
    "validation_fraction": 0.15,
    "n_iter_no_change": 15,
    "alpha": 1e-4,
    "random_state": 42,
}

MODEL_VERSION = "1.2.0-url-only"
DATASET_NAME = "UCI Phishing Websites (Training Dataset.arff)"


def train() -> dict:
    df = load_arff()
    X, y = split_features_target(df)
    data = make_splits(X, y)

    model = MLPClassifier(**MODEL_PARAMS)
    model.fit(data["X_train"], data["y_train"])

    val_pred = model.predict(data["X_val"])
    val_acc = accuracy_score(data["y_val"], val_pred)
    val_f1 = f1_score(data["y_val"], val_pred)
    print(f"Converged in {model.n_iter_} iterations")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Validation F1-score: {val_f1:.4f}")

    joblib.dump(model, MODELS_DIR / "fnn_model.joblib")
    joblib.dump(data["scaler"], MODELS_DIR / "scaler.joblib")
    with open(MODELS_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(data["feature_names"], f, indent=2)
    with open(MODELS_DIR / "model_card.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_version": MODEL_VERSION,
                "dataset": DATASET_NAME,
                "architecture": "MLPClassifier (feedforward NN)",
                "hidden_layers": list(MODEL_PARAMS["hidden_layer_sizes"]),
                "activation": MODEL_PARAMS["activation"],
                "optimizer": MODEL_PARAMS["solver"],
                "learning_rate_init": MODEL_PARAMS["learning_rate_init"],
                "batch_size": MODEL_PARAMS["batch_size"],
                "l2_alpha": MODEL_PARAMS["alpha"],
                "random_state": MODEL_PARAMS["random_state"],
                "n_iter": int(model.n_iter_),
                "feature_names": data["feature_names"],
                "class_mapping": {"0": "Legitimate", "1": "Phishing"},
                "feature_encoding": {
                    "-1": "phishing indicator",
                    "0": "suspicious",
                    "1": "legitimate indicator",
                },
                "label_source": (
                    "UCI 'Result' column: -1 is phishing, +1 is legitimate. "
                    "Training maps Result == -1 to class 1 so that "
                    "predict_proba[:, 1] is the probability of phishing."
                ),
            },
            f,
            indent=2,
        )

    # training loss curve
    loss_curve = list(map(float, model.loss_curve_))
    with open(RESULTS_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump({"loss_curve": loss_curve, "n_iter": int(model.n_iter_)}, f, indent=2)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(loss_curve)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training loss")
        ax.set_title("Training loss")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "training_loss.png", dpi=110)
        plt.close(fig)
    except Exception as exc:
        print(f"(matplotlib unavailable: {exc})")

    print(f"Saved model to {MODELS_DIR/'fnn_model.joblib'}")
    return {"val_accuracy": val_acc, "val_f1": val_f1, "n_iter": int(model.n_iter_)}


if __name__ == "__main__":
    train()
