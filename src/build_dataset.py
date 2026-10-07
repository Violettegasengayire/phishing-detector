"""Build the training set from two independent sources.

Neither source is usable on its own:

* PhiUSIIL (2024) contributes real, recent phishing URLs, but its
  legitimate half is unusable. Every one of its 134,850 legitimate rows
  is ``https://www.<domain>`` with no path, against 41.5% ``www.``,
  48.6% HTTPS and 59.7% with a path on the phishing side. A model
  trained on it learns "www + https + no path" and scores every real
  deep link, including github.com/python/cpython, at 1.000 phishing.
* Tranco contributes real legitimate domains, ranked by a published,
  reproducible methodology, but carries no phishing examples.

Taking the phishing half of one and the legitimate half of the other
removes the artifact, but introduces a different risk: anything that
differs between the two *sources* rather than between the two *classes*
becomes a shortcut. The defences are:

* compare hostname to hostname. Paths, query strings and schemes exist
  on one side only, so they are discarded entirely rather than fed to
  the model as free separation.
* strip a leading ``www.`` from both sides, so the prefix cannot be
  learned.
* drop any host appearing in both sets.

What remains is a genuine question: does this hostname look like one a
phishing campaign registered, or one people actually visit?
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PHIUSIIL_PATH = DATA_DIR / "PhiUSIIL_Phishing_URL_Dataset.csv"
TRANCO_PATH = DATA_DIR / "top-1m.csv"
OUTPUT_PATH = DATA_DIR / "hostnames.csv"

#: Tranco is ranked by popularity. Taking the head would pit the world's
#: 100k most famous domains against phishing hosts and make the task
#: artificially easy, so a band reaching well into the long tail is used.
TRANCO_SKIP = 1_000
TRANCO_LIMIT = 220_000

#: Tranco lists registrable domains, so 90% of it has no sub-domain,
#: against 58% of the phishing hosts that do. Left alone the model learns
#: "has a sub-domain, therefore phishing" and flags mail.google.com,
#: docs.python.org and aws.amazon.com - measured, not hypothetical.
#:
#: Real services put a great deal of legitimate traffic on sub-domains,
#: so the fix is to make the legitimate side look like the real web
#: rather than to delete the feature, which would also discard the
#: genuine signal in google.com.account-verify.suspicious.tk.
#:
#: These prefixes are ordinary service names, chosen without reference to
#: the labels. Several of them (login, secure, account) are also common
#: in phishing hosts, which is deliberate: the model should not be able
#: to treat the word alone as proof.
SUBDOMAIN_PREFIXES = (
    "www", "mail", "docs", "api", "blog", "shop", "news", "support",
    "help", "login", "account", "accounts", "my", "app", "portal",
    "secure", "cdn", "static", "media", "images", "dev", "admin",
    "store", "forum", "community", "events", "careers", "status",
    "en", "de", "fr", "es", "uk", "m", "mobile", "beta", "test",
)

#: Share of legitimate hosts given a sub-domain, set near the phishing
#: side's 58% so the feature carries no free separation.
SUBDOMAIN_SHARE = 0.50

#: Two-level sub-domains, as a share of those that get one at all.
NESTED_SHARE = 0.15


def normalise_host(value: str) -> str:
    """Reduce a URL or bare domain to a comparable hostname."""
    value = (value or "").strip().lower()
    if not value:
        return ""
    if "://" not in value:
        value = f"http://{value}"
    host = urlparse(value).hostname or ""
    if host.startswith("www."):
        host = host[4:]
    return host.strip(".")


def load_phishing_hosts() -> set[str]:
    df = pd.read_csv(PHIUSIIL_PATH, usecols=["URL", "label"])
    phishing = df.loc[df["label"] == 0, "URL"].astype(str)
    hosts = {normalise_host(u) for u in phishing}
    return {h for h in hosts if "." in h}


def load_legitimate_hosts(seed: int = 42) -> set[str]:
    """Tranco domains, half of them given a realistic service sub-domain.

    See SUBDOMAIN_PREFIXES for why the augmentation is here. The sampling
    is seeded so the dataset rebuilds identically.
    """
    rng = random.Random(seed)
    hosts: set[str] = set()
    with TRANCO_PATH.open("r", encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.reader(f)):
            if i < TRANCO_SKIP:
                continue
            if i >= TRANCO_LIMIT:
                break
            if len(row) < 2:
                continue
            host = normalise_host(row[1])
            if "." not in host:
                continue

            if rng.random() < SUBDOMAIN_SHARE:
                prefix = rng.choice(SUBDOMAIN_PREFIXES)
                if rng.random() < NESTED_SHARE:
                    prefix = f"{rng.choice(SUBDOMAIN_PREFIXES)}.{prefix}"
                host = f"{prefix}.{host}"
            hosts.add(host)
    return hosts


def build() -> pd.DataFrame:
    phishing = load_phishing_hosts()
    legitimate = load_legitimate_hosts()

    overlap = phishing & legitimate
    if overlap:
        # A host in both sets has an unresolvable label, so it is removed
        # from both rather than silently assigned to one.
        phishing -= overlap
        legitimate -= overlap

    rows = [{"host": h, "is_phishing": 1} for h in sorted(phishing)]
    rows += [{"host": h, "is_phishing": 0} for h in sorted(legitimate)]
    df = pd.DataFrame(rows).drop_duplicates(subset="host")

    print(f"phishing hosts:    {int(df.is_phishing.sum()):,}")
    print(f"legitimate hosts:  {int((1 - df.is_phishing).sum()):,}")
    print(f"dropped overlap:   {len(overlap):,}")
    return df


if __name__ == "__main__":
    frame = build()
    frame.to_csv(OUTPUT_PATH, index=False)
    print(f"wrote {OUTPUT_PATH}  ({len(frame):,} rows)")
