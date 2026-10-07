"""Integration tests for the production prediction pipeline.

These tests load the trained model and verify that the end-to-end
``PhishingDetector.analyze`` call succeeds on known-good URLs. They
skip gracefully when DNS is unavailable.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from url_features import FEATURE_NAMES               # noqa: E402
from predict import PhishingDetector                 # noqa: E402
from risk import interpret, unverified               # noqa: E402
from url_validator import URLValidationError         # noqa: E402


def _dns_available() -> bool:
    try:
        socket.gethostbyname("www.wikipedia.org")
        return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def detector():
    return PhishingDetector()


class TestModelLoad:
    def test_feature_names_match_extractor(self, detector):
        assert detector.feature_names == list(FEATURE_NAMES)

    def test_model_version_set(self, detector):
        assert detector.model_version != ""


class TestPredictions:
    def test_rejects_private_ip(self, detector):
        with pytest.raises(URLValidationError):
            detector.analyze("http://192.168.1.1/paypal.php")

    def test_rejects_javascript_scheme(self, detector):
        with pytest.raises(URLValidationError):
            detector.analyze("javascript:alert(1)")

    @pytest.mark.skipif(not _dns_available(), reason="network unavailable")
    def test_known_legitimate_is_low_risk(self, detector):
        report = detector.analyze("https://www.wikipedia.org")
        assert report.risk.verdict == "low"
        assert report.probability_phishing < 0.3

    @pytest.mark.skipif(not _dns_available(), reason="network unavailable")
    def test_phishing_looking_url_flags(self, detector):
        # Resolvable host (weebly.com) + several URL signals (dashes,
        # 'paypal', long path) should pull the model away from legitimate.
        # It lands in the "treat with caution" band rather than "high":
        # the only strong phishing feature the URL exposes is the dash in
        # the domain, so the model is appropriately not certain.
        url = "http://secure-login-paypal-verify.weebly.com/update-info.php"
        report = detector.analyze(url, allow_ssl_check=False)
        assert report.probability_phishing > 0.5
        assert report.risk.verdict in ("medium", "high")


class TestFeatureVectorShape:
    def test_feature_count_matches_training(self, detector):
        assert len(detector.feature_names) == len(FEATURE_NAMES)


class TestRiskLabels:
    def test_user_facing_risk_states_are_explicit(self):
        assert interpret(0.95, ["suspicious"], threshold=0.7).title == "Likely Phishing"
        assert interpret(0.5, ["mixed"], threshold=0.7).title == "Possibly Suspicious"
        assert interpret(0.1, ["safe"], threshold=0.7).title == "Likely Legitimate"
        assert unverified("not_found").title == "Unverified / Could Not Be Checked"

    def test_phishing_like_unresolved_domain_is_flagged(self, detector):
        report = detector.analyze("https://google-login-security-example.com")
        assert report.risk.title == "Likely Phishing"
        assert report.probability_phishing is not None
        assert report.probability_phishing >= 0.8
