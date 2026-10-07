"""Tests for the production validation pipeline that runs before the model.

The rule these tests protect: the phishing model only ever sees an address
that passed every check. If the domain cannot be confirmed to exist, the
pipeline stops and reports that it could not check, rather than returning
a verdict it has no basis for.

The bug that motivated this: entering a syntactically valid but
non-existent domain produced "Likely Legitimate Website". A domain that
does not resolve is not safe, and it is not phishing either. It is
unknown, and unknown needs its own answer.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predict import PhishingDetector                       # noqa: E402
from url_validator import (                                # noqa: E402
    ResolutionStatus,
    URLValidationError,
    _resolve,
    validate,
)


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


# Hostnames under .invalid are guaranteed by RFC 2606 never to resolve,
# so these cases do not depend on a particular domain staying unregistered.
NONEXISTENT = [
    "nonexistent-host-for-tests.invalid",
    "http://another-missing-host.invalid/login",
    "https://deep.sub.domain.missing-host.invalid/account/verify",
]

LEGITIMATE = [
    "https://www.wikipedia.org",
    "https://www.google.com",
    "https://github.com",
]

MALFORMED = [
    "",
    "   ",
    "not a url at all",
    "http://",
    "https://.",
    "http://-leadinghyphen.com",
    "http://trailinghyphen-.com",
    "http://double..dot.com",
    "user@example.com",
    "javascript:alert(1)",
    "ftp://files.example.com/x",
    "http://example.com:99999/",
]

PRIVATE_OR_LOCAL = [
    "http://localhost/admin",
    "http://127.0.0.1/",
    "http://10.0.0.1/",
    "http://192.168.1.1/",
    "http://172.16.0.5/",
    "http://169.254.169.254/latest/meta-data/",
    "http://[::1]/",
]


class TestExistingLegitimateDomains:
    """A real domain must reach the model and come back with a prediction."""

    @needs_network
    @pytest.mark.parametrize("url", LEGITIMATE)
    def test_resolves_and_is_analysed(self, detector, url):
        report = detector.analyze(url)
        assert report.was_analysed, f"{url} should have reached the model"
        assert report.probability_phishing is not None
        assert report.resolution == ResolutionStatus.RESOLVED.value
        assert report.risk.verdict in ("low", "medium", "high")

    @needs_network
    @pytest.mark.parametrize("url", LEGITIMATE)
    def test_is_not_reported_as_unverified(self, detector, url):
        assert detector.analyze(url).risk.verdict != "unverified"


class TestNonexistentDomains:
    """The case that motivated this work."""

    @needs_network
    @pytest.mark.parametrize("url", NONEXISTENT)
    def test_returns_neutral_result(self, detector, url):
        report = detector.analyze(url)
        assert report.risk.verdict == "unverified"
        assert report.risk.title == "Website Could Not Be Verified"
        assert report.resolution == ResolutionStatus.NOT_FOUND.value

    @needs_network
    @pytest.mark.parametrize("url", NONEXISTENT)
    def test_model_never_runs(self, detector, url):
        report = detector.analyze(url)
        assert not report.was_analysed
        assert report.probability_phishing is None
        assert report.feature_values == {}

    @needs_network
    @pytest.mark.parametrize("url", NONEXISTENT)
    def test_is_not_called_legitimate_or_phishing(self, detector, url):
        """Requirements 7 and 8: absence of a domain is not evidence
        either way, so neither word may appear in what the user reads."""
        report = detector.analyze(url)
        text = f"{report.risk.title} {report.risk.headline} {report.risk.message}"
        assert "legitimate" not in text.lower()
        assert "phishing" not in text.lower()

    @needs_network
    def test_nonexistent_domain_without_dashes(self, detector):
        """The original report: a short, clean-looking name that happens
        not to exist. It previously scored 0.000 and read as legitimate,
        because the model's strongest feature keys on dashes."""
        report = detector.analyze("fanbrice.invalid")
        assert report.risk.verdict == "unverified"
        assert not report.was_analysed


class TestMalformedDomains:
    """Malformed input is rejected during validation, before any lookup."""

    @pytest.mark.parametrize("url", MALFORMED)
    def test_rejected(self, detector, url):
        with pytest.raises(URLValidationError):
            detector.analyze(url)

    @pytest.mark.parametrize("url", MALFORMED)
    def test_rejection_message_is_safe(self, detector, url):
        """Requirement: no implementation detail leaks to the user."""
        with pytest.raises(URLValidationError) as excinfo:
            detector.analyze(url)
        message = str(excinfo.value).lower()
        for leak in ("dns", "getaddrinfo", "onnx", "fnn", "scaler",
                     "traceback", "socket", "errno", "numpy"):
            assert leak not in message


class TestResolverFailureStates:
    """DNS failure, timeout and network failure are distinct states.

    These patch the resolver rather than relying on a real outage, so the
    behaviour is pinned without making the suite depend on the network.
    """

    def test_timeout_is_its_own_state(self, monkeypatch, detector):
        import url_validator

        def hang(*args, **kwargs):
            raise TimeoutError("resolver did not answer")

        monkeypatch.setattr(url_validator.socket, "getaddrinfo", hang)
        report = detector.analyze("https://example.com")
        # TimeoutError is the same class as concurrent.futures.TimeoutError
        # on 3.11+, so a resolver timeout reports as a timeout either way.
        assert report.resolution == ResolutionStatus.TIMEOUT.value
        assert report.risk.verdict == "unverified"
        assert not report.was_analysed

    def test_slow_resolver_times_out(self, monkeypatch):
        import time
        import url_validator

        def slow(*args, **kwargs):
            time.sleep(3)
            return [(2, 1, 6, "", ("93.184.216.34", 0))]

        monkeypatch.setattr(url_validator.socket, "getaddrinfo", slow)
        status, ips = url_validator._resolve("example.com", timeout=0.3)
        assert status is ResolutionStatus.TIMEOUT
        assert ips == []

    def test_name_not_found_is_distinct_from_network_error(self, monkeypatch):
        import url_validator

        def nxdomain(*args, **kwargs):
            raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")

        monkeypatch.setattr(url_validator.socket, "getaddrinfo", nxdomain)
        status, _ = url_validator._resolve("missing.invalid")
        assert status is ResolutionStatus.NOT_FOUND

        def unreachable(*args, **kwargs):
            raise OSError("network is unreachable")

        monkeypatch.setattr(url_validator.socket, "getaddrinfo", unreachable)
        status, _ = url_validator._resolve("example.com")
        assert status is ResolutionStatus.NETWORK_ERROR

    def test_every_failure_state_blocks_inference(self, monkeypatch, detector):
        import url_validator

        for failure in (
            socket.gaierror(socket.EAI_NONAME, "not known"),
            OSError("network is unreachable"),
        ):
            def raiser(*args, _exc=failure, **kwargs):
                raise _exc

            monkeypatch.setattr(url_validator.socket, "getaddrinfo", raiser)
            report = detector.analyze("https://example.com")
            assert not report.was_analysed
            assert report.risk.verdict == "unverified"


class TestValidPhishingSamples:
    """A resolving host with phishing-shaped features still gets scored.

    The verification gate must not become a way for phishing to escape
    analysis, so these assert the model still runs and still reacts.
    """

    @needs_network
    def test_untrusted_certificate_host_is_analysed_and_flagged(self, detector):
        report = detector.analyze("https://self-signed.badssl.com")
        assert report.was_analysed
        assert report.probability_phishing > 0.5
        assert report.risk.verdict in ("medium", "high")

    @needs_network
    def test_phishing_shaped_url_on_resolving_host_is_analysed(self, detector):
        url = "http://secure-login-paypal-verify.weebly.com/update-info.php"
        report = detector.analyze(url, allow_ssl_check=False)
        assert report.was_analysed
        assert report.probability_phishing > 0.5
        assert report.risk.verdict in ("medium", "high")

    @needs_network
    def test_gate_does_not_suppress_phishing_verdicts(self, detector):
        """A high-risk verdict must still be reachable end to end."""
        report = detector.analyze("https://untrusted-root.badssl.com")
        assert report.was_analysed
        assert report.risk.verdict == "high"


class TestPrivateAndLocalAddresses:
    """SSRF guards: these are refused outright, not merely unverified."""

    @pytest.mark.parametrize("url", PRIVATE_OR_LOCAL)
    def test_rejected(self, detector, url):
        with pytest.raises(URLValidationError):
            detector.analyze(url)

    def test_hostname_resolving_into_private_space_is_rejected(self, monkeypatch):
        """A public name pointing at RFC1918 is the dangerous case: it
        passes syntax checks and only the post-resolution guard stops it."""
        import url_validator

        def private(*args, **kwargs):
            return [(2, 1, 6, "", ("10.1.2.3", 0))]

        monkeypatch.setattr(url_validator.socket, "getaddrinfo", private)
        with pytest.raises(URLValidationError):
            validate("http://internal.example.com")

    @pytest.mark.parametrize("url", PRIVATE_OR_LOCAL)
    def test_never_reaches_the_model(self, detector, url):
        """Rejection must happen during validation, so no inference runs."""
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


class TestPipelineOrdering:
    """The required order: normalise, validate, check existence, then infer."""

    def test_malformed_input_is_rejected_before_any_lookup(self, monkeypatch):
        import url_validator

        looked_up = []
        monkeypatch.setattr(
            url_validator.socket,
            "getaddrinfo",
            lambda *a, **k: looked_up.append(a) or [],
        )
        for bad in ("not a url", "user@example.com", "javascript:alert(1)"):
            with pytest.raises(URLValidationError):
                validate(bad)
        assert looked_up == [], "DNS was queried for input that is not a URL"

    @needs_network
    def test_hostname_is_extracted_and_normalised(self):
        validated = validate("HTTP://WWW.Example.COM/Path")
        assert validated.host == "www.example.com"
        assert validated.scheme == "http"
