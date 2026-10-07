"""Train the experimental hostname models on 2024 data.

This is not the production model. It exists to answer one question:
would a recent dataset beat the 2015 UCI data the shipped detector uses?
The answer is written up in ``results/dataset_comparison.md``; the short
version is that it trades a documented weakness for a worse one.

Two variants are trained so the comparison is fair:

* ``engineered`` - the 23 hand-built hostname features.
* ``ngram`` - those features plus character n-grams of the hostname,
  which is the technique that gives the supplied ONNX model its recall.

Run ``python src/experiments/build_dataset.py`` first.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from url_features import FEATURE_NAMES, extract_host_features, normalise_host

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "hostnames.csv"
OUT_DIR = ROOT / "models" / "experimental"
RESULTS = ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

#: Real hosts from neither source. A model that scores well on the test
#: split but badly here has learned the sources apart, not the classes.
HOLDOUT_LEGIT = [
    "www.google.com", "github.com", "en.wikipedia.org", "docs.python.org",
    "mail.google.com", "accounts.google.com", "login.microsoftonline.com",
    "aws.amazon.com", "www.paypal.com", "www.coca-cola.com", "t-mobile.com",
    "www.hewlett-packard.com", "stackoverflow.com", "support.apple.com",
    "www.bbc.co.uk", "news.ycombinator.com",
]
HOLDOUT_PHISH = [
    "paypal-secure-login.verify-account.tk",
    "secure-appleid.apple.com.verify-login.ru",
    "banking-chase-verify.com", "facebook-login-secure.ml",
    "appleid.apple.com-verify-account-login.secure-update.info",
    "google.com.account-verify.suspicious-domain.xyz",
    "httpsamazon-security.com", "f0519141.xsph.ru", "shprakserf.gq",
    "secure-login-paypal-verify.weebly.com",
]


def _balance(idx_pos, idx_neg, rng):
    n = min(len(idx_pos), len(idx_neg))
    keep = np.concatenate([
        rng.choice(idx_pos, n, replace=False),
        rng.choice(idx_neg, n, replace=False),
    ])
    rng.shuffle(keep)
    return keep


def _metrics(y_true, proba, threshold=0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
    }


def run() -> dict:
    df = pd.read_csv(DATASET).drop_duplicates(subset="host").reset_index(drop=True)
    hosts = df["host"].astype(str)
    X = pd.DataFrame(
        [extract_host_features(h) for h in hosts], columns=FEATURE_NAMES
    ).astype(np.float64)
    y = df["is_phishing"].astype(int)

    X_tr, X_te, y_tr, y_te, h_tr, h_te = train_test_split(
        X, y, hosts, test_size=0.15, random_state=RANDOM_STATE, stratify=y
    )
    rng = np.random.default_rng(RANDOM_STATE)
    keep = _balance(y_tr.index[y_tr == 1], y_tr.index[y_tr == 0], rng)
    X_tr, y_tr, h_tr = X_tr.loc[keep], y_tr.loc[keep], h_tr.loc[keep]

    scaler = StandardScaler().fit(X_tr)
    report: dict = {}

    # ---- variant 1: engineered features only -------------------------
    engineered = MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=200, early_stopping=True,
        n_iter_no_change=12, batch_size=256, random_state=RANDOM_STATE,
    ).fit(scaler.transform(X_tr), y_tr)
    proba = engineered.predict_proba(scaler.transform(X_te))[:, 1]
    report["engineered"] = _metrics(y_te, proba)

    # ---- variant 2: engineered features + character n-grams ----------
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 4), min_df=5,
        max_features=20_000, sublinear_tf=True,
    )
    A_tr = vec.fit_transform(h_tr)
    A_te = vec.transform(h_te)
    F_tr = hstack([csr_matrix(scaler.transform(X_tr)), A_tr]).tocsr()
    F_te = hstack([csr_matrix(scaler.transform(X_te)), A_te]).tocsr()

    ngram = MLPClassifier(
        hidden_layer_sizes=(128, 64), max_iter=60, early_stopping=True,
        n_iter_no_change=8, batch_size=512, random_state=RANDOM_STATE,
    ).fit(F_tr, y_tr)
    proba_n = ngram.predict_proba(F_te)[:, 1]
    report["ngram"] = _metrics(y_te, proba_n)

    # ---- the check that matters: hosts from neither source -----------
    def score_engineered(host: str) -> float:
        row = pd.DataFrame(
            [extract_host_features(host)], columns=FEATURE_NAMES
        ).astype(np.float64)
        return float(engineered.predict_proba(scaler.transform(row))[0, 1])

    def score_ngram(host: str) -> float:
        h = normalise_host(host)
        row = pd.DataFrame(
            [extract_host_features(h)], columns=FEATURE_NAMES
        ).astype(np.float64)
        combined = hstack(
            [csr_matrix(scaler.transform(row)), vec.transform([h])]
        ).tocsr()
        return float(ngram.predict_proba(combined)[0, 1])

    for name, fn in (("engineered", score_engineered), ("ngram", score_ngram)):
        legit = {h: round(fn(h), 3) for h in HOLDOUT_LEGIT}
        phish = {h: round(fn(h), 3) for h in HOLDOUT_PHISH}
        report[name]["holdout_legit"] = legit
        report[name]["holdout_phishing"] = phish
        report[name]["holdout_false_positives"] = sum(v > 0.5 for v in legit.values())
        report[name]["holdout_missed"] = sum(v <= 0.5 for v in phish.values())

    joblib.dump(engineered, OUT_DIR / "hostname_fnn.joblib")
    joblib.dump(ngram, OUT_DIR / "hostname_ngram_fnn.joblib")
    joblib.dump(scaler, OUT_DIR / "hostname_scaler.joblib")
    joblib.dump(vec, OUT_DIR / "hostname_vectorizer.joblib")
    with open(OUT_DIR / "experiment_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    for name in ("engineered", "ngram"):
        r = report[name]
        print(f"{name}:")
        print(f"   test acc={r['accuracy']:.4f}  recall={r['recall']:.4f}  "
              f"f1={r['f1']:.4f}  auc={r['roc_auc']:.4f}")
        print(f"   held-out: {r['holdout_false_positives']}/{len(HOLDOUT_LEGIT)} "
              f"false positives, {r['holdout_missed']}/{len(HOLDOUT_PHISH)} missed")
    print(f"\nWrote {OUT_DIR / 'experiment_report.json'}")
    return report


if __name__ == "__main__":
    run()
