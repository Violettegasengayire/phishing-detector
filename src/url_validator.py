"""URL validation and SSRF guards.

Every URL submitted by a user is untrusted input. Before the feature
extractor is allowed to touch the network, this module checks:

* scheme is http or https
* length is bounded
* the parsed URL has a hostname
* the resolved IP addresses are all public (no loopback, no RFC1918,
  no link-local, no multicast, no cloud-metadata address)

The ``validate`` function returns a ValidatedURL dataclass or raises
``URLValidationError`` with a human message that is safe to show.
"""

from __future__ import annotations

import concurrent.futures
import ipaddress
import re
import socket
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse


MAX_URL_LENGTH = 2048
ALLOWED_SCHEMES = {"http", "https"}

# Addresses we never want the backend to even touch.
_BLOCKED_HOSTNAMES = {
    "localhost", "ip6-localhost", "ip6-loopback", "metadata.google.internal",
}
_METADATA_IPS = {
    "169.254.169.254",   # AWS / GCP / Azure IMDS
    "fd00:ec2::254",     # AWS IPv6 IMDS
}


#: How long to wait for the name resolver before giving up.
DNS_TIMEOUT_SECONDS = 5.0


class URLValidationError(ValueError):
    """Raised when a user-submitted URL cannot be analysed safely."""


class ResolutionStatus(str, Enum):
    """Outcome of the domain existence check.

    These are deliberately distinct. "The domain does not exist",
    "the resolver did not answer in time" and "we could not reach a
    resolver at all" are three different situations, and none of them
    is evidence about whether a site is phishing. Collapsing them into
    one boolean is what lets a non-existent domain be reported as
    legitimate.
    """

    RESOLVED = "resolved"
    NOT_FOUND = "not_found"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"


@dataclass
class ValidatedURL:
    original: str
    normalised: str
    scheme: str
    host: str
    port: int | None
    resolved_ips: list[str] = field(default_factory=list)
    resolution: ResolutionStatus = ResolutionStatus.RESOLVED

    @property
    def is_resolved(self) -> bool:
        return self.resolution is ResolutionStatus.RESOLVED


_DANGEROUS_SCHEME_RE = re.compile(
    r"^(?:javascript|data|file|about|vbscript|mailto|ftp|ssh|telnet|gopher)\s*:",
    re.IGNORECASE,
)

# Reasonably strict email match: local-part @ domain.tld (no scheme prefix).
# This runs on the raw user input before any URL parsing happens, so we
# catch things like "name@gmail.com" that would otherwise look like URL
# userinfo once prefixed with http://.
_EMAIL_RE = re.compile(
    r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$"
)

GENERIC_HELP = "Enter a valid website address, for example: google.com"


def _normalise(url: str) -> str:
    url = url.strip()
    if not url:
        raise URLValidationError("Please enter a website address.")
    if len(url) > MAX_URL_LENGTH:
        raise URLValidationError("That address is too long to analyse.")
    if _DANGEROUS_SCHEME_RE.match(url):
        raise URLValidationError("Only http and https addresses are supported.")
    # Reject email-shaped input before it gets wrapped in http://
    bare = re.sub(r"^(?:https?://)", "", url, flags=re.IGNORECASE)
    if _EMAIL_RE.match(bare):
        raise URLValidationError(GENERIC_HELP)
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", url):
        url = "http://" + url
    return url


def _is_public_ip(ip_text: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_text)
    except ValueError:
        return False
    if ip.is_loopback or ip.is_private or ip.is_link_local:
        return False
    if ip.is_multicast or ip.is_reserved or ip.is_unspecified:
        return False
    if ip_text in _METADATA_IPS:
        return False
    return True


#: gaierror codes that mean "this name has no address record", as opposed
#: to a resolver that is unreachable or slow. EAI_NODATA is absent on some
#: platforms, so it is looked up defensively.
_NAME_NOT_FOUND_CODES = {
    getattr(socket, "EAI_NONAME", None),
    getattr(socket, "EAI_NODATA", None),
    11001,  # WSAHOST_NOT_FOUND (Windows)
    11004,  # WSANO_DATA (Windows)
}
_NAME_NOT_FOUND_CODES.discard(None)


def _resolve(
    host: str, *, timeout: float = DNS_TIMEOUT_SECONDS
) -> tuple[ResolutionStatus, list[str]]:
    """Look up a hostname and report *why* the lookup ended as it did.

    ``getaddrinfo`` has no timeout argument and blocks on the system
    resolver, so it runs on a worker thread that can be abandoned. The
    thread is left to finish on its own; it holds no locks and exits
    when the resolver replies or the OS gives up.
    """
    # Not used as a context manager: __exit__ joins the worker, which would
    # block for the full resolver timeout and undo the deadline above.
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(
            socket.getaddrinfo, host, None, 0, socket.SOCK_STREAM
        )
        try:
            infos = future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return ResolutionStatus.TIMEOUT, []
        except socket.gaierror as exc:
            if exc.errno in _NAME_NOT_FOUND_CODES:
                return ResolutionStatus.NOT_FOUND, []
            return ResolutionStatus.NETWORK_ERROR, []
        except OSError:
            return ResolutionStatus.NETWORK_ERROR, []
    finally:
        pool.shutdown(wait=False)

    ips = sorted({info[4][0] for info in infos})
    if not ips:
        return ResolutionStatus.NOT_FOUND, []
    return ResolutionStatus.RESOLVED, ips


def _is_valid_domain(host: str) -> bool:
    """A DNS hostname must have at least one dot, labels of 1-63 chars
    made of letters/digits/hyphens (not starting or ending with hyphen),
    and a TLD label of at least two letters."""
    if len(host) > 253 or "." not in host:
        return False
    labels = host.split(".")
    if any(not lbl for lbl in labels):
        return False
    label_re = re.compile(r"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)$")
    if not all(label_re.match(lbl) for lbl in labels):
        return False
    tld = labels[-1]
    if len(tld) < 2 or not tld.isalpha():
        return False
    return True


def validate(url: str, *, require_resolution: bool = True) -> ValidatedURL:
    normalised = _normalise(url)
    parsed = urlparse(normalised)

    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise URLValidationError("Only http and https addresses are supported.")

    # Reject embedded credentials. These can be used to disguise the
    # real host from a human reader and should not be passed to the
    # feature extractor.
    if parsed.username is not None or parsed.password is not None:
        raise URLValidationError(GENERIC_HELP)

    host = (parsed.hostname or "").lower()
    if not host:
        raise URLValidationError(GENERIC_HELP)
    if host in _BLOCKED_HOSTNAMES:
        raise URLValidationError("Internal and loopback addresses are not accepted.")

    # If the host is already an IP, check it directly
    try:
        ipaddress.ip_address(host)
        host_is_ip = True
    except ValueError:
        host_is_ip = False

    if host_is_ip and not _is_public_ip(host):
        raise URLValidationError("Private, loopback and metadata IPs are not accepted.")

    if not host_is_ip and not _is_valid_domain(host):
        raise URLValidationError(GENERIC_HELP)

    try:
        port = parsed.port
    except ValueError as exc:
        raise URLValidationError("That address contains an invalid port.") from exc
    if port is not None and (port <= 0 or port > 65535):
        raise URLValidationError("Invalid port number in URL.")

    resolved: list[str] = []
    status = ResolutionStatus.RESOLVED
    if host_is_ip:
        resolved = [host]
    elif require_resolution:
        status, resolved = _resolve(host)
        # The SSRF guard runs on whatever came back. A name that resolves
        # into private space is rejected outright rather than reported as
        # unverified, because that is a deliberate attempt to reach the
        # host running this app.
        for ip_text in resolved:
            if not _is_public_ip(ip_text):
                raise URLValidationError(
                    "That host resolves to an internal network address."
                )

    return ValidatedURL(
        original=url.strip(),
        normalised=normalised,
        scheme=scheme,
        host=host,
        port=port,
        resolved_ips=resolved,
        resolution=status,
    )
