"""Load the UCI phishing website dataset and split it into
train / validation / test sets without leakage.

The production FNN uses only the subset of features that can be
observed from a URL + DNS + a strict TLS handshake. That subset is
defined by ``feature_extraction.PRODUCTION_FEATURES`` and this module
mirrors it exactly so training and inference stay in sync.

A second dataset was evaluated as a possible replacement for this one.
See ``src/experiments/`` and ``results/dataset_comparison.md`` for why
it was not adopted.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from feature_extraction import PRODUCTION_FEATURES


ARFF_PATH = Path(__file__).resolve().parents[1] / "data" / "Training Dataset.arff"


def load_arff(path: Path | str = ARFF_PATH) -> pd.DataFrame:
    """Parse the ARFF file into a DataFrame (no scipy dependency)."""
    path = Path(path)
    columns: list[str] = []
    rows: list[list[int]] = []
    in_data = False
    attr_re = re.compile(r"^@attribute\s+(\S+)", re.IGNORECASE)

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            if not in_data:
                if line.lower().startswith("@data"):
                    in_data = True
                    continue
                m = attr_re.match(line)
                if m:
                    columns.append(m.group(1))
                continue
            parts = [p.strip() for p in line.split(",")]
            rows.append([int(p) for p in parts])

    return pd.DataFrame(rows, columns=columns)


def split_features_target(
    df: pd.DataFrame,
    *,
    target: str = "Result",
    feature_subset: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    features = feature_subset or PRODUCTION_FEATURES
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    X = df[features].copy()
    # UCI encoding: Result == -1 is phishing, Result == +1 is legitimate.
    # Model class 1 must mean phishing so predict_proba[:, 1] is P(phishing).
    y = (df[target] == -1).astype(int)
    return X, y


def make_splits(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> dict:
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    rel_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=rel_val,
        random_state=random_state,
        stratify=y_trainval,
    )

    scaler = StandardScaler().fit(X_train)
    return {
        "X_train": scaler.transform(X_train),
        "X_val": scaler.transform(X_val),
        "X_test": scaler.transform(X_test),
        "y_train": y_train.values,
        "y_val": y_val.values,
        "y_test": y_test.values,
        "feature_names": list(X.columns),
        "scaler": scaler,
    }


if __name__ == "__main__":
    df = load_arff()
    print(f"Dataset shape:  {df.shape}")
    print(f"Features used:  {PRODUCTION_FEATURES}")
    X, y = split_features_target(df)
    print(f"Class counts:   {y.value_counts().to_dict()}  (1 = phishing)")
    print(f"Missing values: {int(X.isna().sum().sum())}")
    print(f"Duplicate rows (full dataset): {int(df.duplicated().sum())}")
