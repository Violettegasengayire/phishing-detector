"""Features computed from a hostname.

``extract_host_features`` is called both to build the training matrix and
to score a live submission, so the two representations cannot drift
apart. An earlier version of this project derived them from separate
code paths and the mismatch went unnoticed because nothing failed
loudly; sharing one definition removes that whole class of bug.

The model reads the hostname only. Paths and query strings are ignored
on purpose: the training data pairs phishing URLs from one source with
legitimate domains from another, and only the phishing side has paths,
so any path feature would separate the two sources rather than the two
classes. Working hostname-to-hostname keeps the comparison fair.

The cost of that choice is stated plainly in the README: a phishing page
hosted on a compromised legitimate domain is invisible to this model,
because the hostname really is legitimate.

Nothing here touches the network. Host existence and certificate checks
live in ``url_validator`` and ``feature_extraction``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import urlparse

SHORTENERS = {
    "bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co", "is.gd", "buff.ly",
    "adf.ly", "bl.ink", "lnkd.in", "shorte.st", "cutt.ly", "rebrand.ly",
    "tiny.cc", "x.co", "db.tt", "qr.ae", "u.to", "j.mp", "soo.gd",
    "clck.ru", "shorturl.at",
}

#: Registries that hand out names at no cost, which is why bulk phishing
#: registrations concentrate in them. This is stated domain knowledge
#: rather than a list mined from the labels, so it cannot memorise the
#: training set.
FREE_TLDS = {"tk", "ml", "ga", "cf", "gq", "xyz", "top", "work", "click",
             "link", "buzz", "rest", "icu", "cyou", "sbs"}

#: Counted as an undifferentiated group. Matching specific brand names
#: is what makes the supplied pretrained model score paypal.com at 0.83
#: and login.microsoftonline.com at 0.97, so no brand appears here.
SENSITIVE_WORDS = (
    "secure", "account", "update", "login", "signin", "verify",
    "confirm", "password", "banking", "webscr", "recover", "unlock",
    "suspend", "billing", "invoice", "wallet", "support", "service",
)

_IPV4 = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
_VOWELS = set("aeiou")

#: Order is part of the model contract.
FEATURE_NAMES = [
    "host_length",
    "n_labels",
    "n_subdomains",
    "longest_label_length",
    "mean_label_length",
    "tld_length",
    "is_free_tld",
    "n_hyphens",
    "hyphen_ratio",
    "n_digits",
    "digit_ratio",
    "has_digit_in_tld_label",
    "host_entropy",
    "vowel_ratio",
    "max_consonant_run",
    "is_ip_host",
    "is_shortener",
    "https_token_in_host",
    "n_sensitive_words",
    "has_hex_escape",
    "has_punycode",
    "n_unique_chars",
    "repeated_char_ratio",
]


def _entropy(text: str) -> float:
    """Shannon entropy over characters.

    Bulk-registered domains draw characters more uniformly than words
    do, which separates ``g7x2ka9qz`` from ``wikipedia`` without needing
    a dictionary.
    """
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _max_consonant_run(text: str) -> int:
    best = run = 0
    for ch in text:
        if ch.isalpha() and ch not in _VOWELS:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def normalise_host(value: str) -> str:
    """Reduce a URL or bare domain to the hostname the model scores."""
    value = (value or "").strip().lower()
    if not value:
        return ""
    if "://" not in value:
        value = f"http://{value}"
    host = urlparse(value).hostname or ""
    if host.startswith("www."):
        host = host[4:]
    return host.strip(".")


def extract_host_features(host: str) -> dict[str, float]:
    """Map a hostname to the model's feature dictionary.

    Accepts a full URL as well as a bare hostname; anything beyond the
    host is discarded.
    """
    host = normalise_host(host)
    labels = [lbl for lbl in host.split(".") if lbl]
    tld = labels[-1] if len(labels) > 1 else ""
    name_part = "".join(labels[:-1]) if len(labels) > 1 else host

    length = len(host) or 1
    n_digits = sum(ch.isdigit() for ch in host)
    n_hyphens = host.count("-")
    letters = [ch for ch in host if ch.isalpha()]
    n_vowels = sum(ch in _VOWELS for ch in letters)

    return {
        "host_length": float(len(host)),
        "n_labels": float(len(labels)),
        "n_subdomains": float(max(len(labels) - 2, 0)),
        "longest_label_length": float(max((len(l) for l in labels), default=0)),
        "mean_label_length": (
            sum(len(l) for l in labels) / len(labels) if labels else 0.0
        ),
        "tld_length": float(len(tld)),
        "is_free_tld": 1.0 if tld in FREE_TLDS else 0.0,
        "n_hyphens": float(n_hyphens),
        "hyphen_ratio": n_hyphens / length,
        "n_digits": float(n_digits),
        "digit_ratio": n_digits / length,
        "has_digit_in_tld_label": 1.0 if any(c.isdigit() for c in tld) else 0.0,
        "host_entropy": _entropy(host),
        "vowel_ratio": (n_vowels / len(letters)) if letters else 0.0,
        "max_consonant_run": float(_max_consonant_run(name_part)),
        "is_ip_host": 1.0 if _IPV4.match(host) else 0.0,
        "is_shortener": 1.0 if host in SHORTENERS else 0.0,
        "https_token_in_host": 1.0 if "https" in host else 0.0,
        "n_sensitive_words": float(sum(w in host for w in SENSITIVE_WORDS)),
        "has_hex_escape": 1.0 if re.search(r"0x[0-9a-f]{2}", host) else 0.0,
        "has_punycode": 1.0 if "xn--" in host else 0.0,
        "n_unique_chars": float(len(set(host))),
        "repeated_char_ratio": 1.0 - (len(set(host)) / length),
    }


def feature_vector(host: str) -> list[float]:
    """The feature dictionary flattened into FEATURE_NAMES order."""
    values = extract_host_features(host)
    return [values[name] for name in FEATURE_NAMES]
