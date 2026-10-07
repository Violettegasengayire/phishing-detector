# Which dataset should the detector use?

The shipped model is trained on the UCI Phishing Websites dataset, which
was collected in 2015. That is old enough to be a fair objection, so a
recent dataset and a supplied pretrained model were both evaluated as
replacements. Neither was adopted. This is the evidence.

Everything below is reproducible:

```
python src/experiments/build_dataset.py
python src/experiments/train_hostname.py
```

Raw numbers are in `models/experimental/experiment_report.json`.

## How the candidates were judged

Test-set accuracy was not treated as the deciding number. Each candidate
was also scored on 26 hosts belonging to neither training source: 16 real
legitimate hosts and 10 phishing-shaped ones. A model that does well on
its own test split but badly here has learned to tell its two data
sources apart rather than to tell phishing from legitimate, and that
difference is invisible in the headline metric.

False positives were weighted more heavily than the accuracy figure
suggests. A detector that flags a real sign-in page teaches its user to
dismiss warnings, which costs more than the warning was ever worth.

## Candidate 1: the supplied `model.onnx`

A TF-IDF text pipeline over raw URL strings feeding a linear classifier
(`Tokenizer`, `StringNormalizer`, `TfidfVectorizer`, `LinearClassifier`,
produced by skl2onnx). Not a neural network, so it does not meet the
assignment's deep-learning requirement, and it takes raw strings rather
than the dataset's feature columns.

Measured on 20 legitimate and 12 phishing URLs:

| | result |
|---|---|
| Phishing caught | 12 / 12 |
| False positives | 5 / 20 |

The false positives are the problem:

| Host | P(phishing) |
|---|---|
| `login.microsoftonline.com` | 0.972 |
| `mail.google.com` | 0.956 |
| `accounts.google.com/signin` | 0.872 |
| `paypal.com` | 0.831 |
| `aws.amazon.com` | 0.695 |

It has learned that strings like `login`, `signin` and `paypal` indicate
phishing. Those words appear in phishing URLs because phishing imitates
sign-in pages, so the correlation is real in the training data and
useless in production: it fires hardest on exactly the pages a user most
needs to trust. Kept in the repository for comparison, not loaded by the
application.

## Candidate 2: UCI PhiUSIIL (2024), used directly

235,795 URLs, published 2024, no missing values. Trained straight out of
the box it reached **99.81%** validation accuracy. That number is an
artifact twice over.

**Leakage.** `URLSimilarityIndex` scores 99.67% on its own with a
depth-3 decision tree. It is computed by comparing each URL against a
reference list of known-legitimate sites, so it encodes the answer.
`TLDLegitimateProb` is a per-TLD legitimacy rate computed across the
whole dataset, which carries test-set information into training. Both
were excluded.

**Collection artifact.** With those removed, accuracy stayed near 99.8%,
because the two halves of the dataset were collected differently:

| | legitimate | phishing |
|---|---|---|
| starts with `www.` | 100.0% | 41.5% |
| HTTPS | 100.0% | 48.6% |
| has a path | 0.0% | 59.7% |

Every one of the 134,850 legitimate URLs is `https://www.<domain>` with
no path. The model learned that shape, not phishing, and scored real
deep links at ceiling:

```
https://github.com/python/cpython       1.000
https://en.wikipedia.org/wiki/Phishing  1.000
https://www.bbc.co.uk/news/world-123    1.000
https://docs.python.org/3/library/os.html 1.000
```

No feature selection fixes this. Removing one artifact moves the model
onto the next, because all three separate the sources perfectly.

## Candidate 3: PhiUSIIL phishing hosts + Tranco legitimate hosts

The phishing half of PhiUSIIL is genuinely useful: 84,018 real, recent
phishing hostnames. Pairing it with legitimate domains from the Tranco
top-1M ranking removes the artifact, at the cost of introducing a new
risk, since anything differing between the two *sources* rather than the
two *classes* becomes a shortcut. Three defences:

* compare hostname to hostname. Paths and schemes exist on one side
  only, so they are discarded rather than handed to the model as free
  separation.
* strip a leading `www.` from both sides.
* drop the 284 hosts appearing in both sets, whose label is unresolvable.

A first attempt still produced 6 false positives out of 16, every one a
sub-domain (`mail.google.com`, `aws.amazon.com`, `docs.python.org`).
Tranco lists registrable domains, so 90% of it has no sub-domain against
58% of phishing hosts that do, and the model learned "sub-domain,
therefore phishing". Deleting the feature would also discard the real
signal in `google.com.account-verify.suspicious.tk`, so the legitimate
side was instead given realistic service sub-domains (`mail`, `docs`,
`login`, `en`, ...) until the rates matched at 55.0% against 57.8%.

Two variants were then trained.

| | engineered | + char n-grams |
|---|---|---|
| Test accuracy | 0.797 | **0.867** |
| Test recall | 0.698 | **0.841** |
| ROC-AUC | 0.855 | **0.948** |
| Held-out false positives | **2 / 16** | 3 / 16 |
| Held-out phishing missed | 1 / 10 | **0 / 10** |

Character n-grams clearly win on every test-set metric. They also
reproduce the supplied model's defect:

| Host | engineered | n-gram |
|---|---|---|
| `login.microsoftonline.com` | 0.270 | **0.774** |
| `www.coca-cola.com` | 0.524 | **0.693** |
| `www.hewlett-packard.com` | 0.632 | 0.637 |

This is the same failure as candidate 1, arrived at independently, which
is the useful part: character n-grams over hostnames learn
brand-adjacent fragments, and the resulting recall is bought by flagging
real sign-in pages. A metric improvement from 0.698 to 0.841 recall is
not worth that, and it would not have been visible without scoring hosts
from outside both sources.

## Decision

The 2015 model ships. It scores 90.8% accuracy and 90.7% phishing recall
on its own test split, and on the 16 held-out legitimate hosts it
produces no false positives at all.

| | shipped (2015) | hostname (2024) | n-gram (2024) | ONNX |
|---|---|---|---|---|
| Test accuracy | **0.908** | 0.797 | 0.867 | n/a |
| Test recall | **0.907** | 0.698 | 0.841 | n/a |
| Held-out false positives | **0** | 2 | 3 | 5 |
| Deep learning | yes | yes | yes | no |

The honest caveat is that those first two columns are not directly
comparable, since each is measured on its own distribution and the 2015
task is the easier one. The column that *is* comparable is the held-out
one, measured identically for all four, and the shipped model is the only
candidate that never flags a real site.

This does not make the 2015 data good. Its weakness is documented in the
README: certificates were expensive when it was collected, so the model
treats a trusted certificate as strong evidence of legitimacy, and
certificates are now free. A phishing site with a valid certificate
scores lower than it should. That is a known, bounded, documented
weakness, and it is preferable to an unknown one or to a model that
calls Microsoft's sign-in page phishing.

## What would actually fix this

Not another feature set. The gap is in the data: a legitimate corpus of
real URLs *with paths*, drawn from the same kind of source as the
phishing set, so that path structure can be used without separating the
sources. Tranco cannot supply that, since it lists domains. Failing
that, features that have not drifted since 2015 and cannot be imitated
cheaply, such as WHOIS domain age, would be more use than any further
tuning of the current inputs.
