"""Phishing Website Checker — production Streamlit interface.

Users paste a web address and press Check Website. The app validates
the URL, extracts safe features, runs the production model and shows a
plain-language verdict. Technical details are available in a separate
collapsible section intended for assessors, not end users.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from predict import AnalysisReport, get_detector   # noqa: E402
from url_validator import URLValidationError       # noqa: E402


st.set_page_config(
    page_title="Phishing Website Checker",
    page_icon="shield",
    layout="centered",
)

# ---- restrained cybersecurity-tool styling -----------------------------
st.markdown(
    """
    <style>
      :root { color-scheme: dark; }
      html, body, [class*="css"] {
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                       Roboto, Helvetica, Arial, sans-serif;
      }
      .block-container { padding-top: 2rem; max-width: 760px; }
      h1 { font-size: 1.5rem !important; font-weight: 600;
           margin-bottom: 0.25rem; color: #f1f5f9; }
      .subtitle { color: #94a3b8; font-size: 0.92rem;
                  margin-bottom: 1.5rem; }
      .result-card {
          border: 1px solid #334155;
          border-radius: 6px;
          padding: 18px 20px;
          margin-top: 1rem;
          background: #1e293b;
      }
      .result-card.high   { border-left: 4px solid #ef4444; }
      .result-card.medium { border-left: 4px solid #f59e0b; }
      .result-card.low    { border-left: 4px solid #22c55e; }
      .result-card.unverified { border-left: 4px solid #64748b; }
      .result-title { font-size: 1.05rem; font-weight: 600;
                      color: #f1f5f9; }
      .result-headline { font-size: 0.82rem; font-weight: 600;
                         letter-spacing: 0.6px; text-transform: uppercase;
                         margin-top: 2px; }
      .result-headline.high   { color: #f87171; }
      .result-headline.medium { color: #fbbf24; }
      .result-headline.low    { color: #4ade80; }
      .result-headline.unverified { color: #94a3b8; }
      .result-message { color: #cbd5e1; font-size: 0.92rem;
                        margin-top: 10px; line-height: 1.5; }
      .reasons-label  { font-weight: 600; font-size: 0.85rem;
                        color: #e2e8f0; margin-top: 14px; margin-bottom: 2px; }
      .reasons        { color: #94a3b8; font-size: 0.88rem;
                        margin: 2px 0 0 1.1rem; padding: 0; }
      .meta           { color: #64748b; font-size: 0.78rem;
                        margin-top: 14px; }
      .footer         { color: #64748b; font-size: 0.78rem;
                        margin-top: 2.5rem; border-top: 1px solid #1e293b;
                        padding-top: 1rem; }
      .stButton>button {
          background: #334155; color: #f1f5f9;
          border: 1px solid #475569; border-radius: 4px;
          padding: 0.55rem 1.1rem;
          font-weight: 500;
      }
      .stButton>button:hover { background: #475569; color: #ffffff;
                               border-color: #64748b; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def _detector():
    return get_detector()


@st.cache_data(show_spinner=False)
def _metrics():
    path = ROOT / "results" / "metrics.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


@st.cache_data(show_spinner=False)
def _model_card():
    path = ROOT / "models" / "model_card.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def _render_result(report: AnalysisReport) -> None:
    r = report.risk
    cls = r.verdict
    st.markdown(
        f"""
        <div class='result-card {cls}'>
          <div class='result-title'>{r.title}</div>
          <div class='result-headline {cls}'>{r.headline}</div>
          <div class='result-message'>{r.message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if r.reasons:
        st.markdown(
            "<div class='reasons-label'>Why this was flagged</div>",
            unsafe_allow_html=True,
        )
        for reason in r.reasons:
            st.markdown(f"- {reason}")


# ---- main view ----------------------------------------------------------
st.markdown("<h1>Phishing Website Checker</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Check a website address for signs commonly "
    "associated with phishing.</div>",
    unsafe_allow_html=True,
)

with st.form(key="url_form", clear_on_submit=False):
    url_input = st.text_input(
        "Website address",
        placeholder="https://example.com",
        label_visibility="visible",
    )
    submitted = st.form_submit_button("Check Website")

if submitted:
    cleaned = url_input.strip() if url_input else ""
    if not cleaned:
        st.warning("Please enter a website address.")
    else:
        detector = _detector()
        try:
            with st.spinner("Checking website..."):
                report = detector.analyze(cleaned)
        except URLValidationError as exc:
            st.error(str(exc))
        except Exception:
            st.error(
                "The checker could not complete the analysis. Please try "
                "again, or try a different address."
            )
        else:
            _render_result(report)


# ---- technical details (for the assessor; not the user) ----------------
with st.expander("About the Detection Model"):
    st.markdown("### Problem")
    st.write(
        "Phishing websites imitate real services to steal passwords, "
        "payment details, or personal information. This tool scores a "
        "web address against patterns observed in a labelled phishing "
        "dataset."
    )

    st.markdown("### Dataset")
    st.write(
        "UCI Phishing Websites dataset: 11,055 labelled samples. Each "
        "sample has 30 categorical features; this tool trains on the "
        "subset of those features that can be obtained safely from a "
        "URL, DNS lookup and TLS handshake, so the training and "
        "production feature representations match exactly."
    )

    st.markdown("### Preprocessing and splits")
    st.write(
        "Stratified 70 / 15 / 15 train / validation / test split with "
        "`random_state=42`. `StandardScaler` is fit on the training "
        "split only and persisted to `models/scaler.joblib`."
    )

    card = _model_card()
    if card:
        st.markdown("### Model")
        st.write(f"**Version:** `{card.get('model_version', 'unknown')}`")
        st.write(
            f"**Architecture:** {card['architecture']} with hidden "
            f"layers {card['hidden_layers']} and {card['activation']} "
            "activation."
        )
        st.write(
            f"**Optimizer:** {card['optimizer']} "
            f"(lr={card['learning_rate_init']}, batch={card['batch_size']}, "
            f"L2={card['l2_alpha']}). Early stopping with patience 15."
        )
        st.write(
            f"**Features used (in order):** `{', '.join(card['feature_names'])}`"
        )
        st.write(f"**Converged in:** {card['n_iter']} epochs")

    m = _metrics()
    if m:
        st.markdown("### Evaluation (held-out test set)")
        st.write(f"- Accuracy: {m['accuracy']*100:.2f}%")
        st.write(f"- Precision (phishing): {m['precision']*100:.2f}%")
        st.write(f"- Recall (phishing): {m['recall']*100:.2f}%")
        st.write(f"- F1-score: {m['f1_score']*100:.2f}%")
        st.write(f"- ROC-AUC: {m['roc_auc']:.4f}")
        cm = m["confusion_matrix"]
        st.write(
            f"- Confusion matrix — TN={cm['true_negatives']}, "
            f"FP={cm['false_positives']}, FN={cm['false_negatives']}, "
            f"TP={cm['true_positives']} "
            f"(test size {m['test_set_size']})"
        )
        cm_png = ROOT / "results" / "confusion_matrix.png"
        if cm_png.exists():
            st.image(str(cm_png), caption="Confusion matrix (test set)")

    st.markdown("### Limitations")
    st.write(
        "- The checker reads the URL, resolves DNS, and performs a TLS "
        "handshake. It does not fetch page content, so it cannot see "
        "the HTML, forms, or JavaScript on the site.\n"
        "- The underlying dataset is from 2015. New phishing techniques "
        "may not resemble the training examples.\n"
        "- A low-risk result is not proof that a website is safe. "
        "Always be cautious with addresses you did not expect."
    )

st.markdown(
    "<div class='footer'>The checker performs read-only safety checks "
    "and never opens the submitted website. Private IPs, loopback hosts, "
    "and non-HTTP schemes are rejected before any network activity.</div>",
    unsafe_allow_html=True,
)
