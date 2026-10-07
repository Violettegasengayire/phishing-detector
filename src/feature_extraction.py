"""Turn a validated URL into model input and user-readable reasons.

Two separate jobs live here:

* the numeric vector the network scores, which comes from
  ``url_features.extract_host_features`` - the same function training
  uses, so the two cannot drift apart.
* the short explanations shown under a warning. These are plain
  observations about the address, written so someone who does not know
  what a TLS handshake is can still act on them.

A TLS check still runs, but its result is no longer a model input. The
2015 dataset treated a trusted certificate as strong evidence of
legitimacy, which stopped being true once certificates became free; the
current model reads the hostname instead. The handshake is kept because
a certificate that fails to verify is worth telling the user about on
its own terms.

Nothing here fetches a page, follows a redirect, or runs JavaScript.
"""

from __future__ import annotations

import socket
import ssl
from dataclasses import dataclass, field

from url_features import (
    FEATURE_NAMES,
    FREE_TLDS,
    SENSITIVE_WORDS,
    SHORTENERS,
    extract_host_features,
    normalise_host,
)
from url_validator import ValidatedURL


@dataclass
class ExtractedFeatures:
    url: str
    host: str
    values: dict[str, float] = field(default_factory=dict)
    signals: list[str] = field(default_factory=list)
    debug: dict[str, str] = field(default_factory=dict)

    def as_vector(self) -> list[float]:
        return [self.values[name] for name in FEATURE_NAMES]


def _tls_state(host: str, port: int, timeout: float = 3.0) -> tuple[str, str]:
    """Report whether the host serves a certificate that verifies.

    Returns one of ``verified``, ``untrusted`` or ``none``. A handshake
    is performed and immediately closed; no request is sent and no
    content is read.
    """
    context = ssl.create_default_context()
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=host):
                return "verified", "verified-chain"
    except (ssl.SSLCertVerificationError, ssl.CertificateError):
        pass
    except (ssl.SSLError, socket.timeout, OSError) as exc:
        return "none", f"no-tls: {exc.__class__.__name__}"

    unverified = ssl.create_default_context()
    unverified.check_hostname = False
    unverified.verify_mode = ssl.CERT_NONE
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            with unverified.wrap_socket(sock, server_hostname=host):
                return "untrusted", "tls-untrusted-issuer"
    except (ssl.SSLError, socket.timeout, OSError) as exc:
        return "none", f"no-tls: {exc.__class__.__name__}"


def _describe(host: str, values: dict[str, float]) -> list[str]:
    """Plain-language observations about the hostname.

    These explain a warning; they are not the model's reasoning, and the
    app only shows them when the verdict is already raised.
    """
    notes: list[str] = []
    if values["is_ip_host"]:
        notes.append("The address uses a raw IP number instead of a domain name.")
    if values["n_hyphens"] >= 2:
        notes.append(
            "The domain name contains several dashes, a pattern common in "
            "addresses set up to imitate a real brand."
        )
    if values["n_subdomains"] >= 2:
        notes.append(
            "The address stacks extra names in front of the real domain, "
            "which can make it look like a site it is not."
        )
    if values["is_free_tld"]:
        notes.append(
            "The domain ends in a suffix that is given away free, which is "
            "where bulk phishing registrations tend to concentrate."
        )
    if values["n_sensitive_words"] >= 1:
        found = [w for w in SENSITIVE_WORDS if w in host]
        notes.append(
            "The domain name itself contains "
            f"\"{found[0]}\", which is a word phishing pages use to look "
            "like a sign-in screen."
        )
    if values["n_digits"] >= 4:
        notes.append("The domain name contains an unusual number of digits.")
    if values["is_shortener"]:
        notes.append(
            "This is a shortened link, so the real destination is hidden "
            "until it opens."
        )
    if values["https_token_in_host"]:
        notes.append(
            "The domain name itself contains \"https\", which is a known "
            "way of making an address look secure."
        )
    if values["has_punycode"]:
        notes.append(
            "The domain uses special characters that can be made to look "
            "like ordinary letters."
        )
    return notes


def extract(
    validated: ValidatedURL, *, allow_tls_check: bool = True
) -> ExtractedFeatures:
    host = normalise_host(validated.host)
    values = extract_host_features(host)
    signals = _describe(host, values)
    debug: dict[str, str] = {"host": host}

    if validated.scheme != "https":
        signals.append(
            "The connection is not encrypted, so anything typed into the "
            "page could be read in transit."
        )
        debug["tls"] = "scheme=http"
    elif allow_tls_check:
        state, note = _tls_state(validated.host, validated.port or 443)
        debug["tls"] = note
        if state == "untrusted":
            signals.append(
                "The site's security certificate could not be verified."
            )
        elif state == "none":
            signals.append(
                "The site did not complete a secure connection."
            )
    else:
        debug["tls"] = "check-skipped"

    return ExtractedFeatures(
        url=validated.normalised, host=host, values=values,
        signals=signals, debug=debug,
    )
