# Phishing Website Checker

A small cybersecurity utility that lets a user paste a web address and
get a plain-language verdict on whether it looks like a phishing site.
Built around a feedforward neural network trained on the UCI Phishing
Websites dataset. The user never has to understand ML.

## Who this is for

* **End users**: paste a URL, press *Check Website*, read the verdict.
* **Assessors**: open the *About the Detection Model* section inside
  the app to see the dataset, preprocessing, model architecture,
  parameters, and evaluation metrics.

## Project layout

```
phishing-detector/
├── app.py                        Streamlit interface
├── requirements.txt
├── README.md
├── data/
│   ├── Training Dataset.arff     UCI Phishing Websites dataset (shipped model)
│   ├── PhiUSIIL_Phishing_URL_Dataset.csv   2024 dataset (comparison only)
│   ├── top-1m.csv                Tranco ranking (comparison only)
│   └── hostnames.csv             combined build (comparison only)
├── models/
│   ├── fnn_model.joblib          trained production FNN
│   ├── scaler.joblib             fitted StandardScaler
│   ├── feature_names.json        exact feature order used at inference
│   ├── model_card.json           model version + parameters
│   ├── pretrained_model.onnx     supplied ONNX model (comparison only)
│   └── experimental/             2024-dataset models + experiment_report.json
├── src/
│   ├── url_validator.py          validation, SSRF guards, domain existence
│   ├── feature_extraction.py     URL -> safe feature extraction
│   ├── preprocessing.py          dataset loading + splits
│   ├── train.py                  trains the FNN
│   ├── evaluate.py               test-set metrics
│   ├── risk.py                   probability -> user-facing message
│   ├── predict.py                PhishingDetector service
│   └── experiments/              dataset comparison, not used by the app
│       ├── build_dataset.py      PhiUSIIL phishing + Tranco legitimate
│       ├── url_features.py       hostname features
│       └── train_hostname.py     trains + scores the 2024 variants
├── tests/
│   ├── test_url_validator.py
│   ├── test_prediction_pipeline.py
│   ├── test_domain_verification.py
│   └── test_qa_production.py
└── results/
    ├── metrics.json
    ├── confusion_matrix.png
    ├── training_loss.png
    └── dataset_comparison.md     why the 2024 data was not adopted
```

Everything under `src/experiments/` is evidence for a decision, not part
of the running application. The app imports nothing from it.

## What the production pipeline does

```
URL ─▶ normalise ─▶ syntax + scheme checks ─▶ hostname ─▶ domain
                            │                              existence
                            │ malformed                    check
                            ▼                                 │
                      rejected, no lookup       ┌─────────────┴──────────────┐
                                                │                            │
                                          not found /                    resolved
                                       timeout / network                      │
                                          error                               ▼
                                                │                   SSRF guard on IPs
                                                ▼                             │
                                  "Could not be verified"          ┌──────────┴────────┐
                                     (model never runs)            │                   │
                                                             private IP            public IP
                                                                   │                   │
                                                                   ▼                   ▼
                                                               rejected      feature extraction
                                                                                       │
                                                                                       ▼
                                                              user result ◀─ risk ◀─ FNN inference
```

Each stage is a separate module. The Streamlit app only orchestrates
them and renders the result.

The model runs only on an address that cleared every stage above it.
That ordering is the point: a verdict is only meaningful if there was
something real to look at.

## URL validation and security

Every submission is treated as untrusted input. Before any network
activity, [`url_validator.py`](src/url_validator.py) checks:

* non-empty, ≤ 2048 characters
* scheme is `http` or `https`; `javascript:`, `data:`, `file:`,
  `ftp:`, `mailto:`, `about:`, `vbscript:`, `ssh:`, `telnet:`,
  `gopher:` are rejected
* hostname is present and is not `localhost`
* resolved IPs are **all public**: loopback (127/8), private (10/8,
  172.16/12, 192.168/16), link-local (169.254/16), multicast, reserved,
  unspecified, and the cloud metadata address (169.254.169.254) are
  rejected
* port, if specified, is a valid number

The feature extractor only reads the URL string, performs a DNS lookup
(already done by the validator), and does a strict TLS handshake. It
never sends an HTTP request, follows a redirect, downloads a page, or
executes JavaScript.

### Domain existence check

After the syntax checks and before any inference, the validator
resolves the hostname and records *why* the lookup ended as it did.
The resolver runs on a worker thread with a 5-second deadline, because
`getaddrinfo` has no timeout of its own and would otherwise hang the
request.

| State | Meaning | Result |
|---|---|---|
| `resolved` | the host has at least one public address | analysis proceeds |
| `not_found` | the name has no address record | *Website Could Not Be Verified* |
| `timeout` | the resolver did not answer in time | *Website Could Not Be Verified* |
| `network_error` | no resolver could be reached | *Website Could Not Be Verified* |

These are kept separate on purpose. "This domain does not exist",
"the resolver was slow" and "this machine is offline" are three
different facts, and none of them says anything about phishing.
Collapsing them into one boolean is what produced the original bug.

**Why unresolvable addresses are not scored.** A syntactically valid
domain that does not exist used to be passed to the model, which
returned *Likely Legitimate Website* for `fanbrice.rw` with probability
0.000. The model had no basis for that: the features it leans on hardest
describe a certificate and a host that were never there, and the score
was really driven by the absence of a dash in the name. Two rules
follow, and both are tested:

* a domain is not legitimate merely because it exists
* a domain is not phishing merely because it does not exist

So the pipeline stops and says it could not check. Refusing to answer
is the honest result when there is nothing to look at.

One consequence worth stating: the `DNSRecord` feature is now always
`+1` whenever the model runs, since the `-1` case is intercepted
earlier. The column is kept so the inference vector still matches the
training columns. This is a deliberate trade, not an oversight.

Addresses that resolve into private or loopback space are a separate
matter and are still **rejected**, not reported as unverified. A public
name pointing at `10.0.0.0/8` is an attempt to make the server reach
its own network, and there is a test for exactly that case.

## Model

The production model is a feedforward neural network (`MLPClassifier`
from scikit-learn). It is trained on the subset of UCI features that
can be observed from a URL + DNS + TLS, so the training distribution
matches what the extractor can produce at inference time. No missing
feature is ever filled with a median or zero at inference.

| Parameter | Value |
|---|---|
| Framework | scikit-learn `MLPClassifier` |
| Hidden layers | `(32, 16)` ReLU |
| Output | sigmoid, binary |
| Optimizer | Adam, learning rate `1e-3` |
| Batch size | 64 |
| L2 regularisation (`alpha`) | `1e-4` |
| Max epochs | 300 (early stopping, patience 15) |
| Validation (training-time) | 15% stratified |
| Seed | 42 |

The exact feature set stored in `models/feature_names.json`:

1. `having_IP_Address`
2. `URL_Length`
3. `Shortining_Service`
4. `having_At_Symbol`
5. `double_slash_redirecting`
6. `Prefix_Suffix`
7. `having_Sub_Domain`
8. `SSLfinal_State` (from TLS handshake)
9. `HTTPS_token`
10. `port`
11. `DNSRecord`

### Why the full 30-feature dataset was not used in production

Nineteen of the thirty UCI features depend on WHOIS records, HTML
scraping, backlink counts, or 2015-era reputation services. Reaching
those at inference time would mean visiting the suspected URL and
hitting external APIs that no longer exist. The earlier version of this
project kept all thirty features and imputed the unknown ones with the
training-set median; test metrics looked impressive but live
predictions were unreliable. The current build drops those features at
training time so training and inference share the same representation.

### The supplied ONNX model

`models/pretrained_model.onnx` is a scikit-learn pipeline exported by
`skl2onnx`: `Tokenizer` → `StringNormalizer` → `TfidfVectorizer` →
`LinearClassifier`. It takes a raw URL string, and its classifier is a
single linear boundary with no hidden layers, so it does not meet the
assignment's deep-learning requirement and cannot read the dataset's
feature columns.

It was still benchmarked rather than dismissed on architecture. It
catches phishing well (12/12) but flags 5 of 20 legitimate sites,
including `login.microsoftonline.com` at 0.972, `mail.google.com` at
0.956 and `paypal.com` at 0.831. It has learned that `login`, `signin`
and brand names indicate phishing, so it fires hardest on the sign-in
pages a user most needs to trust. Retained for comparison, not loaded
by the app.

### Was a more recent dataset tried?

Yes. The UCI PhiUSIIL dataset (2024, 235,795 URLs) was downloaded and
evaluated, along with a combined build pairing its phishing hosts with
legitimate domains from the Tranco top-1M ranking. Neither replaced the
shipped model.

The short version: PhiUSIIL reached 99.81% accuracy through a
combination of target leakage and a collection artifact (all 134,850 of
its legitimate URLs are `https://www.<domain>` with no path), and models
trained on it score real deep links such as
`https://github.com/python/cpython` at 1.000 phishing. The combined
build fixes that but reaches only 0.698 recall, and the character-n-gram
variant that lifts recall to 0.841 reproduces the ONNX model's habit of
flagging `login.microsoftonline.com`.

The full comparison, including the held-out test used to tell real skill
from source-separation, is in
[results/dataset_comparison.md](results/dataset_comparison.md). The code
is under [src/experiments/](src/experiments/) and reproduces with:

```powershell
python src/experiments/build_dataset.py
python src/experiments/train_hostname.py
```

## Evaluation

`python src/evaluate.py` runs the saved model on the held-out test
split and writes `results/metrics.json` plus
`results/confusion_matrix.png`. The current numbers (reproducible,
seeded) are:

| Metric | Value |
|---|---|
| Accuracy | 89.8% |
| Precision (phishing) | 95.0% |
| Recall (phishing) | 66.8% |
| F1-score | 78.4% |
| ROC-AUC | 0.9454 |
| Decision threshold | 0.84 |
| Test set size | 45,410 hosts |

**Interpretation.** Precision is deliberately favoured over recall. At
the calibrated threshold the detector catches about two thirds of
phishing hosts, and when it does raise a warning it is right roughly 95
times in 100. The alternative operating point, threshold 0.5, lifts
recall to 0.84 but flags real sign-in pages such as
`login.microsoftonline.com` - and a detector people learn to dismiss
protects nobody. The 4,188 missed
phishing hosts are the real cost of that choice and are not hidden: the
app never tells a user a site is safe, only that it found no strong
signs.

### Adjustments made during development

The project went through three model generations. Each change below was
made because a measurement forced it, not to chase a number.

**1.0 - all 30 UCI features.** Test metrics reached 96% by imputing the
19 features that cannot be observed without visiting the site. Live URLs
were misclassified because the inference vector was off-distribution.
Retrained on the URL-observable subset only.

**1.2 - three correctness bugs, all silent.**

* *Labels were inverted.* Training mapped `Result == 1` to class 1, but
  UCI encodes `Result == -1` as phishing, so class 1 meant *legitimate*
  while the app read `predict_proba[:, 1]` as P(phishing). Every verdict
  was backwards. No metric showed it, because the test split was
  inverted identically. It was visible only on live URLs:
  `coca-cola.com` was reported as phishing with probability 1.000.
* *Feature signs were inverted.* The extractor used a local "+1 means
  suspicious" convention against the ARFF's `-1` phishing / `0`
  suspicious / `+1` legitimate. Nine of eleven features were written
  with the wrong sign.
* *`DNSRecord` could never fire.* The validator raised an error for any
  host that failed to resolve, so the feature was pinned at one value
  and dead.

**2.0 - replacing the 2015 data.** A 2015 dataset is a fair objection,
so UCI PhiUSIIL (2024) was evaluated. Straight out of the box it scored
99.81%, which was leakage plus a collection artifact; models trained on
it rated `github.com/python/cpython` at 1.000 phishing. Pairing its
phishing hosts with Tranco legitimate hosts fixed that but reached only
0.698 recall.

**3.0 - n-grams, and a threshold mistake worth recording.** Adding
character n-grams lifted recall from 0.698 to 0.841. At the default 0.5
threshold the model flagged `login.microsoftonline.com`, and this was
first written off as the brand-memorisation flaw the supplied ONNX model
has. That was wrong. The scores separate cleanly:

```
highest legitimate host   0.774
lowest phishing host      0.998
```

The problem was the threshold, not the model. It is now calibrated on
the validation split, choosing the most sensitive point that keeps
precision at or above 0.95, which lands at 0.84. The lesson generalises:
a default threshold is a modelling decision, and leaving it unexamined
nearly discarded the better model.

**Why the 2015 model was retired.** Its headline recall was higher, but
measured on a distribution that no longer exists, and it treated a
trusted certificate as strong evidence of legitimacy. That was true when
certificates cost money; they are now free, and phishing sites carry
valid ones. The full comparison, including the held-out test used to
tell real skill from source-separation, is in
[results/dataset_comparison.md](results/dataset_comparison.md). The 2015
pipeline is preserved under [src/experiments/](src/experiments/).

## How to run

### 1. Install dependencies

```powershell
cd D:\violette\phishing-detector
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Train the model

```powershell
python src/train.py
```

This writes `models/fnn_model.joblib`, `models/scaler.joblib`,
`models/feature_names.json`, and `models/model_card.json`.

### 3. Evaluate

```powershell
python src/evaluate.py
```

Writes `results/metrics.json` and `results/confusion_matrix.png`.

### 4. Run the tests

```powershell
python -m pytest tests/ -v
```

### 5. Launch the app

```powershell
python -m streamlit run app.py
```

Streamlit opens the app in your default browser. Paste a URL and
press **Check Website**.

### CLI prediction (for scripts)

```powershell
python src/predict.py "https://www.wikipedia.org"
```

## Limitations and responsible use

* **It misses about a third of phishing.** At the calibrated threshold
  recall is 66.8%, so roughly one phishing host in three is not flagged.
  This is a deliberate trade for precision: see the evaluation section.
  The app never reports a site as "safe", only that it found no strong
  signs, because a *Low Risk* result genuinely is not proof of anything.
* **The model reads the hostname, not the page or the path.** A phishing
  page hosted on a compromised legitimate domain
  (`https://realsite.com/phish/login`) is invisible to it, because the
  hostname really is legitimate. Nothing in the URL after the host
  changes the score. This is a consequence of the training data: the
  legitimate and phishing hosts come from different sources, and only
  one of them has paths, so any path feature would separate the sources
  rather than the classes.
* **Legitimate hyphenated domains can be flagged.** Phishing hosts are
  much more hyphenated than legitimate ones, so real domains with
  several dashes score high. `hewlett-packard.com` is flagged, and
  `coca-cola.com` lands in *Treat With Caution*. Measured on 14
  well-known legitimate sites, one is a false positive.
* **It does not fetch page content.** DNS is resolved and a TLS
  handshake performed, nothing more. It cannot see forms, scripts or
  iframes, and it never opens or downloads from the submitted address.
* **New phishing techniques may not resemble the training data.** The
  phishing hosts date from 2024. A campaign using a registration style
  that did not exist then may score low.
* **Certificate state is shown, not scored.** A certificate that fails
  to verify is reported to the user, but it is not a model input.
  Certificates are free and automatic, so phishing sites carry valid
  ones and the signal no longer separates the classes.

### What would reduce the risk

The largest single gap is the missing third of phishing. Closing it
needs signals this data does not carry: domain registration age from
WHOIS, which is hard for an attacker to fake, or a legitimate corpus of
real URLs *with paths* so that path structure could be used without
separating the two sources. Further tuning of the current inputs is not
likely to help; that was tested and is written up in
[results/dataset_comparison.md](results/dataset_comparison.md).

For a user, the practical mitigation is the one the interface already
states: treat a clear result as one input, not as permission. Check the
address bar, and do not enter a password on a page reached from a link
you did not expect.
