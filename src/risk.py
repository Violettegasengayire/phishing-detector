"""Convert a phishing probability + extracted feature signals into a
user-facing risk result. The text here is what the end user actually
reads."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RiskResult:
    verdict: str          # "high" | "medium" | "low"
    title: str
    headline: str
    message: str
    reasons: list[str] = field(default_factory=list)


#: Fallback bands, used only if the model card carries no calibrated
#: threshold. The live thresholds are derived from that value below.
HIGH_THRESHOLD = 0.70
LOW_THRESHOLD = 0.30

#: Width of the "treat with caution" band below the decision threshold.
#: Scores just under the threshold are not confidently safe, so they get
#: a hedged answer rather than a clean pass.
CAUTION_MARGIN = 0.25


def unverified(reason: str = "") -> RiskResult:
    """The neutral outcome used when the address could not be checked.

    This is not a third risk level. It means the pipeline stopped before
    the model ran, so there is no prediction to report. Saying "we could
    not check" is the only honest answer: a domain that does not resolve
    is not thereby safe, and it is not thereby phishing either.

    ``reason`` is an internal note for logs and is never shown.
    """
    return RiskResult(
        verdict="unverified",
        title="Unverified / Could Not Be Checked",
        headline="Unverified",
        message=(
            "We could not confirm this address is active or valid. Check "
            "the website address and try again."
        ),
        reasons=[],
    )


def interpret(
    probability_phishing: float,
    signals: list[str],
    *,
    threshold: float | None = None,
) -> RiskResult:
    """Pick the user-facing wording for a prediction.

    ``threshold`` is the model's calibrated decision point, chosen on
    validation data at training time. At or above it the result is
    reported as phishing; within CAUTION_MARGIN below it the result is
    hedged; below that it is reported as likely legitimate.
    """
    reasons = list(dict.fromkeys(signals))   # de-duplicate while preserving order

    high = HIGH_THRESHOLD if threshold is None else threshold
    low = LOW_THRESHOLD if threshold is None else max(high - CAUTION_MARGIN, 0.05)

    if probability_phishing >= high:
        return RiskResult(
            verdict="high",
            title="Likely Phishing",
            headline="Likely Phishing",
            message=(
                "This website shows strong signs commonly associated with "
                "phishing. Avoid entering passwords, payment details, or "
                "personal information."
            ),
            reasons=reasons,
        )
    if probability_phishing <= low:
        return RiskResult(
            verdict="low",
            title="Likely Legitimate",
            headline="Likely Legitimate",
            message=(
                "We did not detect strong phishing indicators. Automated "
                "checks can make mistakes, so remain careful with unfamiliar "
                "websites."
            ),
            reasons=reasons,
        )
    return RiskResult(
        verdict="medium",
        title="Possibly Suspicious",
        headline="Possibly Suspicious",
        message=(
            "The checks produced mixed signals. If you did not expect this "
            "address from someone you trust, do not enter personal "
            "information or sign in."
        ),
        reasons=reasons,
    )
