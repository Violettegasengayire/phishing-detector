"""Unit tests for the URL validator."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from url_validator import URLValidationError, validate   # noqa: E402


def _validate_no_dns(url: str):
    return validate(url, require_resolution=False)


class TestSchemeAndFormat:
    def test_rejects_blank(self):
        with pytest.raises(URLValidationError):
            _validate_no_dns("")

    def test_rejects_javascript_scheme(self):
        with pytest.raises(URLValidationError):
            _validate_no_dns("javascript:alert(1)")

    def test_rejects_data_scheme(self):
        with pytest.raises(URLValidationError):
            _validate_no_dns("data:text/html,<script>")

    def test_rejects_file_scheme(self):
        with pytest.raises(URLValidationError):
            _validate_no_dns("file:///etc/passwd")

    def test_rejects_ftp_scheme(self):
        with pytest.raises(URLValidationError):
            _validate_no_dns("ftp://example.com/")

    def test_rejects_overlong(self):
        with pytest.raises(URLValidationError):
            _validate_no_dns("http://example.com/" + "a" * 3000)

    def test_adds_scheme_when_missing(self):
        v = _validate_no_dns("example.com/path")
        assert v.scheme == "http"
        assert v.host == "example.com"

    def test_accepts_https(self):
        v = _validate_no_dns("https://example.com")
        assert v.scheme == "https"


class TestSSRFGuards:
    def test_rejects_localhost(self):
        with pytest.raises(URLValidationError):
            _validate_no_dns("http://localhost/")

    def test_rejects_loopback_ip(self):
        with pytest.raises(URLValidationError):
            _validate_no_dns("http://127.0.0.1/")

    def test_rejects_private_ip(self):
        with pytest.raises(URLValidationError):
            _validate_no_dns("http://192.168.1.1/")

    def test_rejects_link_local(self):
        with pytest.raises(URLValidationError):
            _validate_no_dns("http://169.254.169.254/latest/meta-data/")

    def test_rejects_rfc1918_10_range(self):
        with pytest.raises(URLValidationError):
            _validate_no_dns("http://10.0.0.1/")


class TestPortHandling:
    def test_rejects_invalid_port_string(self):
        with pytest.raises(URLValidationError):
            _validate_no_dns("http://javascript:alert(1)")  # noqa: parsed as broken port


class TestNonURLInput:
    """Inputs that are plainly not web addresses must be rejected
    without ever reaching the model."""

    def test_rejects_email_address(self):
        with pytest.raises(URLValidationError):
            _validate_no_dns("fabriceraymond53@gmail.com")

    def test_rejects_email_with_scheme(self):
        with pytest.raises(URLValidationError):
            _validate_no_dns("http://fabriceraymond53@gmail.com")

    def test_rejects_bare_word(self):
        with pytest.raises(URLValidationError):
            _validate_no_dns("hello")

    def test_rejects_single_label(self):
        with pytest.raises(URLValidationError):
            _validate_no_dns("google")

    def test_rejects_at_only(self):
        with pytest.raises(URLValidationError):
            _validate_no_dns("@gmail.com")

    def test_rejects_embedded_credentials(self):
        with pytest.raises(URLValidationError):
            _validate_no_dns("http://user:pass@evil.com/path")

    def test_rejects_whitespace_only(self):
        with pytest.raises(URLValidationError):
            _validate_no_dns("   ")

    def test_rejects_hostname_with_underscore(self):
        with pytest.raises(URLValidationError):
            _validate_no_dns("bad_label.example")

    def test_rejects_hostname_trailing_hyphen(self):
        with pytest.raises(URLValidationError):
            _validate_no_dns("example-.com")

    def test_rejects_hostname_leading_hyphen(self):
        with pytest.raises(URLValidationError):
            _validate_no_dns("-example.com")

    def test_rejects_numeric_tld(self):
        with pytest.raises(URLValidationError):
            _validate_no_dns("example.123")


class TestValidPlainInput:
    """Reasonable user input shapes should be accepted and normalised."""

    def test_bare_domain_is_accepted(self):
        v = _validate_no_dns("google.com")
        assert v.scheme == "http"
        assert v.host == "google.com"

    def test_www_domain_is_accepted(self):
        v = _validate_no_dns("www.google.com")
        assert v.host == "www.google.com"

    def test_https_domain_is_accepted(self):
        v = _validate_no_dns("https://google.com")
        assert v.scheme == "https"
        assert v.host == "google.com"

    def test_path_is_preserved(self):
        v = _validate_no_dns("https://accounts.google.com/login")
        assert v.host == "accounts.google.com"
