"""End-to-end QA for the production pipeline (v3.0.0, hostname + n-gram).

Four groups:

1. Legitimate URLs reach the model and are not flagged.
2. Invalid and SSRF-sensitive input is rejected before any inference.
3. The training representation and the inference representation agree.
   A silent mismatch here is the bug that made an earlier version of
   this project report every verdict backwards.
4. The decision threshold is the calibrated one, not the 0.5 default.
   At 0.5 this model flags login.microsoftonline.com, so a regression
   that loses the threshold would be quietly harmful.
"""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from feature_extraction import extract                      # noqa: E402
from predict import PhishingDetector                        # noqa: E402
from preprocessing import build_feature_frame, load_dataset  # noqa: E402
from url_features import (                                   # noqa: E402
    FEATURE_NAMES,
    extract_host_features,
    normalise_host,
)
from url_validator import URLValidationError, validate       # noqa: E402


def _dns_available() -> bool:
    try:
        socket.gethostbyname("www.wikipedia.org")
        return True
    except OSError:
        return False


HAS_NETWORK = _dns_available()
needs_network = pytest.mark.skipif(not HAS_NETWORK, reason="network unavailable")


@pytest.fixture(scope="module")
def detector():
    return PhishingDetector()


@pytest.fixture(scope="module")
def model_card():
    with open(ROOT / "models" / "model_card.json", "r", encoding="utf-8") as f:
        return json.load(f)


LEGIT_URLS = [
    "https://www.wikipedia.org",
    "https://www.google.com",
    "https://github.com",
    "https://docs.python.org/3/library/os.html",
    "https://mail.google.com",
    "https://en.wikipedia.org/wiki/Phishing",
    "https://support.apple.com",
    "https://news.ycombinator.com",
]

INVALID_INPUT = [
    ("empty", ""),
    ("whitespace", "   "),
    ("prose", "not a url at all"),
    ("email", "someone@example.com"),
    ("javascript", "javascript:alert(1)"),
    ("ftp", "ftp://files.example.com/x"),
    ("no host", "http://"),
    ("bad port", "http://example.com:99999/"),
]

SSRF_TARGETS = [
    "http://localhost/admin",
    "http://127.0.0.1/",
    "http://10.0.0.1/",
    "http://192.168.1.1/",
    "http://169.254.169.254/latest/meta-data/",
]


class TestLegitimateURLs:
    @needs_network
    @pytest.mark.parametrize("url", LEGIT_URLS)
    def test_reaches_model_and_is_not_flagged(self, detector, url):
        report = detector.analyze(url)
        assert report.was_analysed, f"{url} never reached the model"
        assert report.risk.verdict != "high", (
            f"{url} => p={report.probability_phishing:.3f}"
        )

    @needs_network
    def test_sign_in_hostnames_are_not_flagged(self, detector):
        """The failure mode of the supplied ONNX model, and of this one
        at the default threshold: real sign-in pages scored as phishing.
        A detector that flags these teaches users to ignore it."""
        for url in (
            "https://login.microsoftonline.com",
            "https://accounts.google.com",
            "https://www.paypal.com",
        ):
            report = detector.analyze(url)
            assert report.risk.verdict != "high", (
                f"{url} flagged at p={report.probability_phishing:.3f}"
            )


class TestInvalidInput:
    @pytest.mark.parametrize("kind,bad", INVALID_INPUT)
    def test_rejected(self, detector, kind, bad):
        with pytest.raises(URLValidationError):
            detector.analyze(bad)

    @pytest.mark.parametrize("kind,bad", INVALID_INPUT)
    def test_message_leaks_no_internals(self, detector, kind, bad):
        with pytest.raises(URLValidationError) as excinfo:
            detector.analyze(bad)
        message = str(excinfo.value).lower()
        for leak in ("dns", "getaddrinfo", "onnx", "fnn", "scaler", "ngram",
                     "tfidf", "traceback", "socket", "errno", "numpy"):
            assert leak not in message

    @pytest.mark.parametrize("url", SSRF_TARGETS)
    def test_ssrf_targets_rejected(self, detector, url):
        with pytest.raises(URLValidationError):
            detector.analyze(url)

    @pytest.mark.parametrize("url", SSRF_TARGETS)
    def test_ssrf_never_reaches_model(self, detector, url):
        calls = []
        original = detector.model.predict_proba
        detector.model.predict_proba = lambda *a, **k: (
            calls.append(1) or original(*a, **k)
        )
        try:
            with pytest.raises(URLValidationError):
                detector.analyze(url)
        finally:
            detector.model.predict_proba = original
        assert calls == []


class TestRepresentationConsistency:
    """Training and inference must build the same vector."""

    def test_detector_columns_match_feature_module(self, detector):
        assert detector.feature_names == list(FEATURE_NAMES)

    def test_saved_feature_names_match_module(self, model_card):
        assert model_card["feature_names"] == list(FEATURE_NAMES)

    @needs_network
    def test_extractor_output_matches_direct_call(self):
        """The live extractor must produce exactly what training would
        produce for the same hostname."""
        validated = validate("https://docs.python.org/3/library/os.html")
        live = extract(validated, allow_tls_check=False).values
        direct = extract_host_features(normalise_host("docs.python.org"))
        assert live == direct

    def test_path_does_not_change_the_vector(self):
        """The model is hostname-scoped. If a path ever altered the
        vector, the scope documented in the README would be wrong."""
        bare = extract_host_features(normalise_host("https://example.com"))
        deep = extract_host_features(
            normalise_host("https://example.com/a/b/c?d=e")
        )
        assert bare == deep


class TestLabelOrientation:
    """An inverted label is silent: every metric stays healthy while the
    app reports phishing sites as safe. These check against facts that
    do not depend on the model."""

    def test_phishing_is_class_one(self):
        # Sampled, not head(): the file is written phishing-first, so a
        # head slice contains one class only.
        df = load_dataset().sample(40_000, random_state=0)
        X, y = build_feature_frame(df)
        assert set(y.unique()) <= {0, 1}
        # Phishing hosts are measurably more hyphenated and more often on
        # free TLDs than legitimate ones.
        assert X.loc[y == 1, "n_hyphens"].mean() > X.loc[y == 0, "n_hyphens"].mean()
        assert X.loc[y == 1, "is_free_tld"].mean() > X.loc[y == 0, "is_free_tld"].mean()

    @needs_network
    def test_obvious_phishing_scores_above_obvious_legitimate(self, detector):
        phishing = detector.analyze(
            "http://secure-login-paypal-verify.weebly.com/x",
            allow_ssl_check=False,
        )
        legitimate = detector.analyze("https://www.wikipedia.org")
        assert phishing.probability_phishing > legitimate.probability_phishing


class TestThresholdCalibration:
    def test_threshold_came_from_the_model_card(self, detector, model_card):
        assert detector.threshold == pytest.approx(
            model_card["decision_threshold"]
        )

    def test_threshold_is_not_the_default(self, detector):
        """0.5 is wrong for this model; n-grams push some legitimate
        sign-in hostnames into the 0.6-0.8 band. If this ever reads 0.5,
        calibration has been lost."""
        assert detector.threshold > 0.5

    def test_reported_metrics_match_the_saved_model(self):
        """metrics.json must describe the model actually on disk."""
        with open(ROOT / "results" / "metrics.json", "r", encoding="utf-8") as f:
            metrics = json.load(f)
        with open(ROOT / "models" / "model_card.json", "r", encoding="utf-8") as f:
            card = json.load(f)
        assert metrics["model_version"] == card["model_version"]
        assert metrics["decision_threshold"] == pytest.approx(
            card["decision_threshold"]
        )
