"""Load the hostname dataset and build training splits.

The data is produced by ``build_dataset.py``, which pairs real phishing
hostnames from UCI PhiUSIIL (2024) with real legitimate hostnames from
the Tranco ranking. See that module for why neither source works alone
and what is done to stop the model separating the sources instead of the
classes.

Features come from ``url_features.extract_host_features``, the same
function the live app calls, so the training and inference
representations are identical by construction. An earlier version of
this project derived them from two separate code paths and the mismatch
went unnoticed because nothing failed loudly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from url_features import FEATURE_NAMES, extract_host_features

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATASET_PATH = DATA_DIR / "hostnames.csv"


def load_dataset(path: Path | str = DATASET_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found. Run 'python src/build_dataset.py' first."
        )
    df = pd.read_csv(path)
    return df.drop_duplicates(subset="host").reset_index(drop=True)


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Compute the feature matrix and verify the label orientation.

    Class 1 is phishing, so ``predict_proba[:, 1]`` reads as the
    probability of phishing. The check below is cheap insurance against
    a silent inversion, which leaves every metric looking healthy while
    the live app reports phishing sites as safe.
    """
    rows = [extract_host_features(h) for h in df["host"].astype(str)]
    X = pd.DataFrame(rows, columns=FEATURE_NAMES).astype(np.float64)
    # X is built positionally, so y must be re-indexed to match. Without
    # this, any caller passing a filtered or sampled frame gets silently
    # misaligned rows.
    y = df["is_phishing"].astype(int).reset_index(drop=True)

    if X.loc[y == 1, "n_hyphens"].mean() <= X.loc[y == 0, "n_hyphens"].mean():
        raise AssertionError(
            "Label orientation check failed: hosts marked phishing are not "
            "more hyphenated than those marked legitimate. Verify the "
            "dataset's label encoding before training."
        )
    return X, y


def _balance(X: pd.DataFrame, y: pd.Series, seed: int):
    """Downsample the majority class to parity, returning the kept index.

    The sources give roughly 84k phishing hosts against 219k legitimate
    ones. Trained on that ratio the network drifts toward the majority
    and validation recall falls to 0.50, which is the expensive half of
    this trade: a false positive costs a few seconds of doubt, a false
    negative hands the user to a phishing page.

    Only the training split is balanced. Validation and test keep the
    natural ratio so the reported numbers describe the real mix.
    """
    rng = np.random.default_rng(seed)
    idx_pos = y.index[y == 1]
    idx_neg = y.index[y == 0]
    n = min(len(idx_pos), len(idx_neg))
    keep = np.concatenate([
        rng.choice(idx_pos, n, replace=False),
        rng.choice(idx_neg, n, replace=False),
    ])
    rng.shuffle(keep)
    return keep


def make_splits(
    X: pd.DataFrame,
    y: pd.Series,
    hosts: pd.Series,
    *,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> dict:
    """Stratified train / validation / test split.

    Hostnames travel alongside the numeric features because the n-gram
    vectoriser is fitted on the training hostnames only.
    """
    X_tv, X_test, y_tv, y_test, h_tv, h_test = train_test_split(
        X, y, hosts, test_size=test_size,
        random_state=random_state, stratify=y,
    )
    rel_val = val_size / (1.0 - test_size)
    X_tr, X_val, y_tr, y_val, h_tr, h_val = train_test_split(
        X_tv, y_tv, h_tv, test_size=rel_val,
        random_state=random_state, stratify=y_tv,
    )

    keep = _balance(X_tr, y_tr, random_state)
    X_tr, y_tr, h_tr = X_tr.loc[keep], y_tr.loc[keep], h_tr.loc[keep]

    scaler = StandardScaler().fit(X_tr)
    return {
        "X_train": X_tr, "y_train": y_tr, "hosts_train": h_tr,
        "X_val": X_val, "y_val": y_val, "hosts_val": h_val,
        "X_test": X_test, "y_test": y_test, "hosts_test": h_test,
        "scaler": scaler,
        "feature_names": list(X.columns),
    }


def load_splits(path: Path | str = DATASET_PATH) -> dict:
    df = load_dataset(path)
    X, y = build_feature_frame(df)
    return make_splits(X, y, df["host"].astype(str))


if __name__ == "__main__":
    df = load_dataset()
    X, y = build_feature_frame(df)
    print(f"Rows:           {len(df):,}")
    print(f"Features:       {X.shape[1]}")
    print(f"Phishing:       {int(y.sum()):,}  ({y.mean():.1%})")
    print(f"Legitimate:     {int((1 - y).sum()):,}")
    print(f"Missing values: {int(X.isna().sum().sum())}")
