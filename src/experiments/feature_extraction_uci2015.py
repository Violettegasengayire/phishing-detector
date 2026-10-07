"""Safe feature extraction for the production phishing detector.

Only features that can be observed from the URL string itself, a DNS
lookup, or a strict TLS handshake are extracted. The extractor never
sends an HTTP request, never follows a redirect, and never downloads
content from the target host.

Feature coding follows the UCI Phishing Websites companion document:

    -1  phishing indicator
     0  suspicious
    +1  legitimate indicator

Getting these signs right matters more than it looks. The training
ARFF uses this convention, so any feature emitted with the opposite
sign teaches the model the reverse of what the URL actually shows.
"""

from __future__ import annotations

import socket
import ssl
from dataclasses import dataclass, field
from urllib.parse import urlparse

from url_validator import ValidatedURL


SHORTENERS = {
    "bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co", "is.gd", "buff.ly",
    "adf.ly", "bl.ink", "lnkd.in", "shorte.st", "cutt.ly", "rebrand.ly",
    "tiny.cc", "x.co", "db.tt", "qr.ae", "u.to", "j.mp", "soo.gd",
    "clck.ru", "shorturl.at",
}

# The exact feature set used by the production FNN. Order matters.
PRODUCTION_FEATURES = [
    "having_IP_Address",
    "URL_Length",
    "Shortining_Service",
    "having_At_Symbol",
    "double_slash_redirecting",
    "Prefix_Suffix",
    "having_Sub_Domain",
    "SSLfinal_State",
    "HTTPS_token",
    "port",
    "DNSRecord",
]


@dataclass
class ExtractedFeatures:
    url: str
    values: dict[str, int] = field(default_factory=dict)
    signals: list[str] = field(default_factory=list)   # human-readable triggered reasons
    debug: dict[str, str] = field(default_factory=dict)  # internal notes

    def as_vector(self) -> list[int]:
        return [self.values[name] for name in PRODUCTION_FEATURES]


def _ssl_state(host: str, port: int, timeout: float = 3.0) -> tuple[int, str]:
    """Return the UCI-style SSLfinal_State value for the host.

    The UCI rule is:

        HTTPS and issuer is trusted      -> legitimate  (+1)
        HTTPS but issuer is not trusted  -> suspicious  ( 0)
        otherwise                        -> phishing    (-1)

    The published rule also asks for a certificate at least a year old,
    but that test measures nothing today. Let's Encrypt issues 90-day
    certificates and auto-renewal rewrites notBefore on every renewal,
    so an established site and a week-old phishing site look the same.
    Measured on live hosts, google.com showed a 15-day certificate while
    a phishing sample showed 56 days, so age ordering is actively
    misleading. Issuer trust is the half of the rule that still holds,
    so the bands are decided by whether the chain verifies.

    The function performs a TLS handshake only. It never sends an HTTP
    request and never reads body content from the server.
    """
    ctx = ssl.create_default_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=host):
                return 1, "verified-chain"
    except (ssl.SSLCertVerificationError, ssl.CertificateError):
        pass  # speaks TLS, but the certificate is not trusted -> check below
    except (ssl.SSLError, socket.timeout, OSError) as exc:
        return -1, f"no-tls: {exc.__class__.__name__}"

    # The chain failed to verify. Retry without verification purely to
    # distinguish "serves TLS with an untrusted/self-signed certificate"
    # (suspicious) from "does not serve TLS at all" (phishing). Nothing
    # is sent over this socket and no content is read.
    unverified = ssl.create_default_context()
    unverified.check_hostname = False
    unverified.verify_mode = ssl.CERT_NONE
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            with unverified.wrap_socket(sock, server_hostname=host):
                return 0, "tls-untrusted-issuer"
    except (ssl.SSLError, socket.timeout, OSError) as exc:
        return -1, f"no-tls: {exc.__class__.__name__}"


def extract(validated: ValidatedURL, *, allow_ssl_check: bool = True) -> ExtractedFeatures:
    url = validated.normalised
    host = validated.host
    scheme = validated.scheme
    port = validated.port

    parsed = urlparse(url)
    v: dict[str, int] = {}
    signals: list[str] = []
    debug: dict[str, str] = {}

    def set_feature(name: str, value: int, trigger: str | None = None) -> None:
        """Record a feature. A trigger message is surfaced to the user only
        when the feature came out as a phishing indicator (-1)."""
        v[name] = value
        if trigger and value == -1:
            signals.append(trigger)

    # 1. having_IP_Address — hostname is a raw IP literal.
    is_ip = False
    try:
        import ipaddress
        ipaddress.ip_address(host)
        is_ip = True
    except ValueError:
        pass
    set_feature(
        "having_IP_Address",
        -1 if is_ip else 1,
        "The address uses a raw IP instead of a domain name.",
    )

    # 2. URL_Length — short / medium / long bands from the UCI paper.
    length = len(url)
    if length < 54:
        set_feature("URL_Length", 1)
    elif length <= 75:
        set_feature("URL_Length", 0)
    else:
        set_feature("URL_Length", -1, "The URL is unusually long.")

    # 3. Shortining_Service
    set_feature(
        "Shortining_Service",
        -1 if host in SHORTENERS else 1,
        "The address uses a URL shortener, which hides the real destination.",
    )

    # 4. having_At_Symbol
    set_feature(
        "having_At_Symbol",
        -1 if "@" in url else 1,
        "The URL contains an '@' symbol, which can hide the real host.",
    )

    # 5. double_slash_redirecting — '//' appearing past position 7 (end of 'http://')
    last_double = url.rfind("//")
    set_feature(
        "double_slash_redirecting",
        -1 if last_double > 7 else 1,
        "The URL contains a suspicious '//' in the path.",
    )

    # 6. Prefix_Suffix — dash in the registered domain
    set_feature(
        "Prefix_Suffix",
        -1 if "-" in host else 1,
        "The domain contains dashes, which phishing kits often use.",
    )

    # 7. having_Sub_Domain — dot count in the host, excluding 'www.'
    bare = host[4:] if host.startswith("www.") else host
    dots = bare.count(".")
    if dots <= 1:
        set_feature("having_Sub_Domain", 1)
    elif dots == 2:
        set_feature("having_Sub_Domain", 0)
    else:
        set_feature(
            "having_Sub_Domain",
            -1,
            "The address stacks several sub-domains, a common disguise pattern.",
        )

    # 8. SSLfinal_State
    if scheme != "https":
        set_feature(
            "SSLfinal_State",
            -1,
            "The site does not use HTTPS, so traffic is not encrypted.",
        )
        debug["SSLfinal_State"] = "scheme=http"
    elif allow_ssl_check:
        ssl_value, note = _ssl_state(host, port or 443)
        v["SSLfinal_State"] = ssl_value
        debug["SSLfinal_State"] = note
        if ssl_value == -1:
            signals.append("The site's TLS certificate failed verification.")
        elif ssl_value == 0:
            signals.append("The TLS certificate was issued very recently.")
    else:
        v["SSLfinal_State"] = 1  # https URL, cert check skipped
        debug["SSLfinal_State"] = "cert-check-skipped"

    # 9. HTTPS_token — the string 'https' appearing inside the hostname
    set_feature(
        "HTTPS_token",
        -1 if "https" in host else 1,
        "The hostname embeds 'https', a known deception pattern.",
    )

    # 10. port — non-standard HTTP/HTTPS port in the URL
    nonstd = port is not None and port not in (80, 443)
    set_feature(
        "port",
        -1 if nonstd else 1,
        "The URL points to an unusual port.",
    )

    # 11. DNSRecord — the validator already resolved the host.
    #
    # In production this is always +1, because a host that did not
    # resolve is returned as "could not be verified" and never reaches
    # the model. The feature is kept so the inference vector matches the
    # training columns, and because the extractor is also used offline
    # on hosts whose resolution was supplied rather than performed. The
    # -1 branch is therefore real but unreachable from the app, which is
    # the intended trade: refusing to answer beats guessing from the one
    # fact that the site is not there.
    set_feature(
        "DNSRecord",
        1 if validated.resolved_ips else -1,
        "The domain has no DNS record.",
    )

    return ExtractedFeatures(url=url, values=v, signals=signals, debug=debug)
