# QA Report

**Case suite:** 30/30 passed  (0 failed).  
**Network tests:** enabled

## Aggregate metrics on held-out test split

- Accuracy: **89.57%**
- Precision (phishing): 88.04%
- Recall (phishing): 94.05%
- F1-score: 90.95%
- Confusion matrix: TN=617 FP=118 FN=55 TP=869 (n=1659)

## Case-level results

| # | Group | Input | Validation | Model prediction | Expected | Result |
|---|---|---|---|---|---|---|
| 1 | valid_legit | `https://github.com` | accepted | low (p=0.004) | low | **PASS** |
| 2 | valid_legit | `http://github.com` | accepted | low (p=0.001) | low | **PASS** |
| 3 | valid_legit | `google.com` | accepted | low (p=0.001) | low | **PASS** |
| 4 | valid_legit | `www.wikipedia.org` | accepted | low (p=0.001) | low | **PASS** |
| 5 | valid_legit | `https://microsoft.com` | accepted | low (p=0.004) | low | **PASS** |
| 6 | valid_legit | `https://en.wikipedia.org/wiki/Phishing` | accepted | low (p=0.006) | low | **PASS** |
| 7 | valid_legit | `https://www.google.com/search?q=python` | accepted | low (p=0.004) | low | **PASS** |
| 8 | invalid | `(empty)` | rejected: Please enter a website address. | model never called | rejected | **PASS** |
| 9 | invalid | `   ` | rejected: Please enter a website address. | model never called | rejected | **PASS** |
| 10 | invalid | `fabriceraymond53@gmail.com` | rejected: Enter a valid website address, for example: google.com | model never called | rejected | **PASS** |
| 11 | invalid | `http://user@gmail.com` | rejected: Enter a valid website address, for example: google.com | model never called | rejected | **PASS** |
| 12 | invalid | `hello` | rejected: Enter a valid website address, for example: google.com | model never called | rejected | **PASS** |
| 13 | invalid | `google` | rejected: Enter a valid website address, for example: google.com | model never called | rejected | **PASS** |
| 14 | invalid | `http://localhost/` | rejected: Internal and loopback addresses are not accepted. | model never called | rejected | **PASS** |
| 15 | invalid | `http://127.0.0.1/` | rejected: Private, loopback and metadata IPs are not accepted. | model never called | rejected | **PASS** |
| 16 | invalid | `http://192.168.1.1/` | rejected: Private, loopback and metadata IPs are not accepted. | model never called | rejected | **PASS** |
| 17 | invalid | `http://10.0.0.1/` | rejected: Private, loopback and metadata IPs are not accepted. | model never called | rejected | **PASS** |
| 18 | invalid | `http://169.254.169.254/latest/meta-data/` | rejected: Private, loopback and metadata IPs are not accepted. | model never called | rejected | **PASS** |
| 19 | invalid | `file:///etc/passwd` | rejected: Only http and https addresses are supported. | model never called | rejected | **PASS** |
| 20 | invalid | `ftp://example.com/` | rejected: Only http and https addresses are supported. | model never called | rejected | **PASS** |
| 21 | invalid | `javascript:alert(1)` | rejected: Only http and https addresses are supported. | model never called | rejected | **PASS** |
| 22 | invalid | `data:text/html,<script>` | rejected: Only http and https addresses are supported. | model never called | rejected | **PASS** |
| 23 | invalid | `mailto:someone@example.com` | rejected: Only http and https addresses are supported. | model never called | rejected | **PASS** |
| 24 | invalid | `bad_label.example` | rejected: Enter a valid website address, for example: google.com | model never called | rejected | **PASS** |
| 25 | invalid | `-example.com` | rejected: Enter a valid website address, for example: google.com | model never called | rejected | **PASS** |
| 26 | invalid | `example.123` | rejected: Enter a valid website address, for example: google.com | model never called | rejected | **PASS** |
| 27 | invalid | `http://user:pass@evil.example/` | rejected: Enter a valid website address, for example: google.com | model never called | rejected | **PASS** |
| 28 | invalid | `http://aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa...` | rejected: That address is too long to analyse. | model never called | rejected | **PASS** |
| 29 | phish_https | `https://secure-login-paypal-verify.weebly.com/account/update` | accepted | high (p=1.000) | high | **PASS** |
| 30 | phish_subtle | `http://a1b2-c3d4-e5f6-g7h8.weebly.com/x/y/z/ident.html` | accepted | high (p=0.996) | medium_or_high | **PASS** |