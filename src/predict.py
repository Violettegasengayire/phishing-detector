"""Production prediction service.

One ``PhishingDetector`` instance holds the loaded model, scaler and
feature-name vector, and exposes a single ``analyze(url)`` entry point
that an application layer can call.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd

from scipy.sparse import csr_matrix, hstack

from feature_extraction import extract
from risk import RiskResult, interpret, unverified
from url_features import FEATURE_NAMES, SENSITIVE_WORDS, extract_host_features, normalise_host
from url_validator import (
    ResolutionStatus,
    URLValidationError,
    ValidatedURL,
    validate,
)


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"


logger = logging.getLogger("phishing_detector")


@dataclass
class AnalysisReport:
    url: str
    risk: RiskResult
    #: ``None`` when the pipeline stopped before the model ran. Callers
    #: must check ``was_analysed`` rather than assuming a number is here.
    probability_phishing: float | None
    feature_values: dict[str, int] = field(default_factory=dict)
    model_version: str = ""
    debug: dict[str, str] = field(default_factory=dict)
    resolution: str = ResolutionStatus.RESOLVED.value

    @property
    def was_analysed(self) -> bool:
        """True when the model actually produced a prediction."""
        return self.probability_phishing is not None


class PhishingDetector:
    """Load the production FNN once and expose a single analyse() call."""

    def __init__(self, models_dir: Path | str = MODELS_DIR) -> None:
        models_dir = Path(models_dir)
        self.model = joblib.load(models_dir / "fnn_model.joblib")
        self.scaler = joblib.load(models_dir / "scaler.joblib")
        self.vectorizer = joblib.load(models_dir / "vectorizer.joblib")
        with open(models_dir / "feature_names.json", "r", encoding="utf-8") as f:
            self.feature_names: list[str] = json.load(f)

        card_path = models_dir / "model_card.json"
        card = {}
        if card_path.exists():
            with open(card_path, "r", encoding="utf-8") as f:
                card = json.load(f)
        self.model_version = card.get("model_version", "unknown")
        # Calibrated on the validation split at training time. The 0.5
        # default is wrong for this model: n-grams push some legitimate
        # sign-in hostnames into the 0.6-0.8 band.
        self.threshold: float = float(card.get("decision_threshold", 0.5))

        # Inference columns must match training columns. An inverted or
        # reordered vector is silent otherwise.
        if list(self.feature_names) != list(FEATURE_NAMES):
            raise RuntimeError(
                "Feature mismatch: the saved model expects "
                f"{self.feature_names} but the extractor produces {FEATURE_NAMES}."
            )

    def _model_input(self, values: dict[str, float], host: str):
        """Build the sparse row the network scores."""
        dense = pd.DataFrame(
            [[values[name] for name in self.feature_names]],
            columns=self.feature_names,
        )
        return hstack([
            csr_matrix(self.scaler.transform(dense)),
            self.vectorizer.transform([host]),
        ]).tocsr()

    # ------------------------------------------------------------------ API
    def analyze(
        self,
        raw_url: str,
        *,
        allow_ssl_check: bool = True,
    ) -> AnalysisReport:
        """Validate the URL, extract features and return a risk result.

        Raises ``URLValidationError`` with a safe message if the URL
        cannot be processed.
        """
        validated: ValidatedURL = validate(raw_url)

        # Gate: the model only runs on an address that passed every
        # production check. If the domain could not be confirmed to
        # exist there is nothing meaningful to score, and guessing
        # either way would be worse than saying so. The features the
        # model relies on most (certificate state, host shape) are
        # unobservable or meaningless for a host that is not there.
        if not validated.is_resolved:
            host = normalise_host(validated.host)
            values = extract_host_features(host)
            matches = [word for word in SENSITIVE_WORDS if word in host]
            phishing_signals: list[str] = []
            if matches:
                phishing_signals.append(
                    "The hostname contains common phishing terms such as "
                    f"{', '.join(matches[:3])}."
                )
            if any(token in host for token in ("login", "secure", "verify", "account", "signin", "password", "update", "confirm", "bank")):
                phishing_signals.append(
                    "The domain name imitates a trusted service and uses login or security wording."
                )
            if "-" in host:
                phishing_signals.append(
                    "The address combines brand-like terms with hyphens, a common phishing pattern."
                )

            if phishing_signals:
                logger.info(
                    "analysis flagged as phishing-like despite unresolved domain",
                    extra={
                        "host": validated.host,
                        "resolution": validated.resolution.value,
                        "matches": matches,
                    },
                )
                risk = interpret(0.95, phishing_signals, threshold=self.threshold)
                return AnalysisReport(
                    url=validated.normalised,
                    risk=risk,
                    probability_phishing=0.95,
                    feature_values=dict(values),
                    model_version=self.model_version,
                    debug={"resolution": validated.resolution.value, "heuristic": "phishing_like_host"},
                    resolution=validated.resolution.value,
                )

            logger.info(
                "analysis skipped: domain not verified",
                extra={
                    "host": validated.host,
                    "resolution": validated.resolution.value,
                },
            )
            return AnalysisReport(
                url=validated.normalised,
                risk=unverified(validated.resolution.value),
                probability_phishing=None,
                feature_values={},
                model_version=self.model_version,
                debug={"resolution": validated.resolution.value},
                resolution=validated.resolution.value,
            )

        features = extract(validated, allow_tls_check=allow_ssl_check)
        proba = float(
            self.model.predict_proba(
                self._model_input(features.values, features.host)
            )[0, 1]
        )

        risk = interpret(proba, features.signals, threshold=self.threshold)
        # The low-risk branch does not need "reasons" shown to the user.
        if risk.verdict == "low":
            risk.reasons = []

        # Structured log, never logs the raw URL to avoid leaking submissions
        logger.info(
            "analysis complete",
            extra={
                "host": validated.host,
                "scheme": validated.scheme,
                "probability": round(proba, 4),
                "verdict": risk.verdict,
                "model_version": self.model_version,
            },
        )

        return AnalysisReport(
            url=validated.normalised,
            risk=risk,
            probability_phishing=proba,
            feature_values=dict(features.values),
            model_version=self.model_version,
            debug=features.debug,
            resolution=validated.resolution.value,
        )


_singleton: PhishingDetector | None = None


def get_detector() -> PhishingDetector:
    """Process-wide singleton loader."""
    global _singleton
    if _singleton is None:
        _singleton = PhishingDetector()
    return _singleton


# ------------------------------------------------------------------ CLI
if __name__ == "__main__":
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else "https://www.github.com"
    detector = get_detector()
    try:
        report = detector.analyze(url)
    except URLValidationError as exc:
        print(f"Rejected: {exc}")
        sys.exit(1)
    print(f"URL: {report.url}")
    print(f"Verdict: {report.risk.title} ({report.risk.headline})")
    if report.was_analysed:
        print(f"Probability phishing: {report.probability_phishing:.3f}")
    else:
        print(f"Not analysed (domain check: {report.resolution})")
    print(f"Model version: {report.model_version}")
    if report.risk.reasons:
        print("Reasons:")
        for r in report.risk.reasons:
            print(f"  - {r}")
