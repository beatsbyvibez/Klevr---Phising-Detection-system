"""
app.py
======
AI-Powered Phishing URL Detection System
Klevr FYP — Olubo Demilade (22/11227)
Caleb University, Imota, Lagos
Supervisor: Prof. M. K. Aregbesola
"""

import os
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import extract_features, feature_names, FEATURE_DESCRIPTIONS
from logger import log_prediction, load_log, clear_log
from web_analyzer import analyze_page, CHECK_DESCRIPTIONS  # Layer 2 module
from blacklist import Blacklist                             # Layer 0 module
from allowlist import check as allowlist_check              # known-good list
from advisor import build_advisory, typewriter_stream, STEPS  # AI-style advisor

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Klevr",
    page_icon="K",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* === Typography === */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"], .stMarkdown, .stTextInput, .stButton {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* === Tighter top padding === */
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px; }

    /* === Page titles: a touch lighter, with subtle bottom border === */
    h1 { font-weight: 700; letter-spacing: -0.02em; }
    h2, h3 { font-weight: 600; letter-spacing: -0.01em; }

    /* === Sidebar refinements === */
    [data-testid="stSidebar"] {
        border-right: 1px solid #334155;
    }
    [data-testid="stSidebar"] .stRadio > label > div {
        font-size: 0.95rem;
    }

    /* === Verdict banners — calmer, no gradient === */
    .verdict-safe, .verdict-phishing, .verdict-confirmed {
        padding: 18px 22px;
        border-radius: 10px;
        font-size: 1.05rem;
        font-weight: 600;
        margin: 4px 0 8px 0;
        border: 1px solid;
    }
    .verdict-safe {
        background: rgba(16, 185, 129, 0.08);
        border-color: rgba(16, 185, 129, 0.4);
        color: #34d399;
    }
    .verdict-phishing {
        background: rgba(239, 68, 68, 0.08);
        border-color: rgba(239, 68, 68, 0.4);
        color: #f87171;
    }
    .verdict-confirmed {
        background: rgba(220, 38, 38, 0.12);
        border-color: rgba(220, 38, 38, 0.6);
        color: #fca5a5;
    }
    .verdict-confirmed .source-label {
        display: inline-block;
        margin-top: 6px;
        font-size: 0.85rem;
        font-weight: 500;
        opacity: 0.85;
    }

    /* === Section headings === */
    .section-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 28px 0 12px 0;
    }

    /* === Buttons === */
    .stButton button {
        font-weight: 500;
        border-radius: 8px;
        transition: all 0.15s ease;
    }
    .stButton button[kind="primary"] {
        background: #3b82f6;
        border: none;
    }
    .stButton button[kind="primary"]:hover {
        background: #2563eb;
        transform: translateY(-1px);
    }

    /* === Inputs === */
    .stTextInput input, .stTextArea textarea {
        border-radius: 8px !important;
        border: 1px solid #334155 !important;
        font-size: 0.95rem !important;
    }
    .stTextInput input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 1px #3b82f6 !important;
    }

    /* === Metric cards (Streamlit native) === */
    [data-testid="stMetric"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 18px;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.78rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f1f5f9;
    }

    /* === Tables === */
    [data-testid="stTable"] table {
        background: transparent !important;
    }
    [data-testid="stTable"] th {
        background: #1e293b !important;
        color: #94a3b8 !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
    }

    /* === Progress bar accent === */
    .stProgress > div > div > div { background: #3b82f6; }

    /* === Footer caption text === */
    .caption {
        font-size: 0.8rem;
        color: #64748b;
    }

    /* === Hide Streamlit branding for a cleaner demo === */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    cols_path  = os.path.join(MODEL_DIR, "feature_cols.pkl")
    if not os.path.exists(model_path):
        return None, None
    model = joblib.load(model_path)
    cols  = joblib.load(cols_path) if os.path.exists(cols_path) else feature_names()
    return model, cols

model, feature_cols = load_model()


# ─────────────────────────────────────────────
# Blacklist (Layer 0) — cached so it survives Streamlit reruns
# ─────────────────────────────────────────────
@st.cache_resource
def load_blacklist():
    """Load the on-disk blacklist cache and refresh it from the feeds
    if more than six hours have passed since the last successful refresh."""
    bl = Blacklist()
    bl.refresh_if_stale()  # silent — never blocks the app even if offline
    return bl

blacklist = load_blacklist()


# ─────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────
def predict_url(url: str):
    """Return (label_str, confidence_phishing, confidence_safe, features_dict)."""
    feats = extract_features(url)
    row   = pd.DataFrame([[feats[c] for c in feature_cols]], columns=feature_cols)
    proba = model.predict_proba(row)[0]
    conf_safe     = float(proba[0])
    conf_phishing = float(proba[1])
    label = "Phishing" if conf_phishing >= 0.5 else "Safe"
    return label, conf_phishing, conf_safe, feats


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 8px 0 16px 0;">
            <div style="font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em; color: #f1f5f9;">
                Klevr
            </div>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 2px;">
                URL threat detection
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio("Navigation", [
        "Single URL Analysis",
        "Batch URL Analysis",
        "Model Performance",
        "Prediction Log",
        "About",
    ], label_visibility="collapsed")
    st.markdown("---")
    # ── Layer 0 blacklist status ──
    n_urls = len(blacklist.urls)
    n_hosts = len(blacklist.hosts)
    st.markdown(
        '<div style="font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;">'
        'Blacklist (Layer 0)</div>',
        unsafe_allow_html=True,
    )
    if n_urls == 0:
        st.warning("Cache is empty. Click refresh below.")
    else:
        col_a, col_b = st.columns(2)
        col_a.metric("URLs", f"{n_urls:,}")
        col_b.metric("Hosts", f"{n_hosts:,}")
        st.caption("Sources: PhishTank · URLhaus")

    if st.button("Refresh blacklist", use_container_width=True):
        with st.spinner("Updating feeds…"):
            status = blacklist.refresh_if_stale(force=True)
        if status.get("refreshed"):
            st.success(f"Done. {status['urls_in_cache']:,} URLs cached.")
        else:
            st.warning(status['reason'])

    st.markdown("---")
    st.markdown(
        '<div style="font-size: 0.75rem; color: #64748b; line-height: 1.6;">'
        'Final Year Project<br>'
        '<b style="color: #94a3b8;">Olubo Demilade</b>, 22/11227<br>'
        'Caleb University, Lagos<br>'
        'Supervisor: Prof. M. K. Aregbesola'
        '</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# Model not found banner
# ─────────────────────────────────────────────
if model is None:
    st.error(
        "**Model not found.**  Run `python train_model.py --dataset your_data.csv` "
        "to train the model first, then restart the app."
    )
    st.stop()


# ═══════════════════════════════════════════════════════════
# PAGE 1 — SINGLE URL ANALYSIS
# ═══════════════════════════════════════════════════════════
if page == "Single URL Analysis":
    st.markdown("# URL Analysis")
    st.markdown(
        '<p style="color: #94a3b8; font-size: 0.95rem; margin-top: -8px; margin-bottom: 28px;">'
        'Paste a URL to check it against the blacklist, the ML model, and (optionally) '
        'a live page scan.</p>',
        unsafe_allow_html=True,
    )

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        url_input = st.text_input(
            "URL", placeholder="https://example.com/login",
            label_visibility="collapsed"
        )
    with col_btn:
        analyse = st.button("Analyse", use_container_width=True, type="primary")

    # Layer 2 toggle — when ticked, also fetch the page and run content checks.
    deep_scan = st.checkbox(
        "Run Layer 2 deep page scan  ·  fetches the live page and inspects "
        "scripts, forms, redirects, and iframes",
        value=False,
    )

    if analyse and url_input.strip():
        url_clean = url_input.strip()
        import time  # used for small pauses between progress steps

        # ── PROGRESS STRIP ──
        # We render a status placeholder that we rewrite as each layer completes,
        # so the user sees the system "thinking" through each step.
        st.markdown(
            '<div class="section-title">Analysis Process</div>',
            unsafe_allow_html=True,
        )
        progress_box = st.empty()

        def _render_progress(states: dict):
            """states maps step_key -> 'running' | 'done' | 'skipped' | 'pending', with optional result text."""
            rows = []
            step_keys = ["allowlist", "blacklist", "features", "model", "layer2"]
            step_labels = {
                "allowlist": "Checking the known-good allowlist",
                "blacklist": "Checking PhishTank and URLhaus blacklists",
                "features":  "Extracting 26 structural features from the URL",
                "model":     "Running the Random Forest model",
                "layer2":    "Running Layer 2 deep page scan",
            }
            total_steps = len(step_keys)
            for i, key in enumerate(step_keys, start=1):
                state = states.get(key, ("pending", ""))
                if isinstance(state, tuple):
                    status, detail = state
                else:
                    status, detail = state, ""
                if status == "done":
                    icon = '<span style="color:#34d399;">●</span>'
                    text_color = "#cbd5e1"
                elif status == "running":
                    icon = '<span style="color:#3b82f6;">◐</span>'
                    text_color = "#f1f5f9"
                elif status == "skipped":
                    icon = '<span style="color:#64748b;">○</span>'
                    text_color = "#64748b"
                else:  # pending
                    icon = '<span style="color:#475569;">○</span>'
                    text_color = "#64748b"
                detail_html = (
                    f'<span style="color:#94a3b8; margin-left:auto;">{detail}</span>'
                    if detail else ""
                )
                rows.append(
                    f'<div style="display:flex; align-items:center; gap:10px; '
                    f'font-size:13px; padding:6px 0; color:{text_color};">'
                    f'{icon}'
                    f'<span style="color:#94a3b8;">Step {i} / {total_steps}</span>'
                    f'<span style="flex:1;">{step_labels[key]}</span>'
                    f'{detail_html}'
                    f'</div>'
                )
            progress_box.markdown("".join(rows), unsafe_allow_html=True)

        # Initial render with all steps pending
        states = {k: "pending" for k in ["allowlist", "blacklist", "features", "model", "layer2"]}
        _render_progress(states)

        # --- Step 1: known-good allowlist ---
        states["allowlist"] = "running"
        _render_progress(states); time.sleep(0.15)
        al_result = allowlist_check(url_clean)
        if al_result["listed"]:
            states["allowlist"] = ("done", f"matched {al_result['matched_domain']}")
        else:
            states["allowlist"] = ("done", "not listed")
        _render_progress(states); time.sleep(0.1)

        # --- Step 2: blacklist ---
        states["blacklist"] = "running"
        _render_progress(states); time.sleep(0.15)
        bl_result = blacklist.check(url_clean)
        bl_detail = (
            f"listed on {bl_result['source']}" if bl_result["listed"] else "not listed"
        )
        states["blacklist"] = ("done", bl_detail)
        _render_progress(states); time.sleep(0.1)

        # --- Step 3 + 4: features + model ---
        # We always extract features (so the advisor has data to talk about)
        # but if the URL is on the allowlist we trust that verdict over the model.
        try:
            states["features"] = "running"
            _render_progress(states); time.sleep(0.1)
            feats = extract_features(url_clean)
            states["features"] = ("done", "26 features extracted")

            states["model"] = "running"
            _render_progress(states); time.sleep(0.1)
            row = pd.DataFrame([[feats[c] for c in feature_cols]], columns=feature_cols)
            proba = model.predict_proba(row)[0]
            raw_conf_safe = float(proba[0])
            raw_conf_phishing = float(proba[1])

            # Final verdict logic, in priority order:
            #   1. Allowlist hit  -> Safe     (confidence floor 0.99)
            #      The allowlist is curated and trusted, so it overrides
            #      any downstream signal. This protects against poisoned
            #      blacklist entries (e.g. a feed that has wrongly listed
            #      a known-good domain) and against model false positives.
            #   2. Blacklist hit  -> Phishing (confidence floor 0.99)
            #   3. Otherwise      -> trust the model
            if al_result["listed"]:
                label = "Safe"
                conf_safe = max(raw_conf_safe, 0.99)
                conf_phishing = 1 - conf_safe
            elif bl_result["listed"]:
                label = "Phishing"
                conf_phishing = max(raw_conf_phishing, 0.99)
                conf_safe = 1 - conf_phishing
            else:
                conf_phishing = raw_conf_phishing
                conf_safe = raw_conf_safe
                label = "Phishing" if conf_phishing >= 0.5 else "Safe"

            states["model"] = ("done", f"{round(conf_phishing*100, 1)}% phishing")
            _render_progress(states); time.sleep(0.1)
            log_prediction(url_clean, label, conf_phishing, conf_safe)
        except Exception as e:
            st.error(f"Error during analysis: {e}")
            st.stop()

        # --- Step 5: Layer 2 (only if checkbox is on) ---
        layer2 = None
        if deep_scan:
            states["layer2"] = "running"
            _render_progress(states); time.sleep(0.1)
            layer2 = analyze_page(url_clean)
            if layer2.get("success"):
                states["layer2"] = ("done", f"suspicion {layer2['suspicion_score']} / 4")
            else:
                states["layer2"] = ("done", "page unreachable")
            _render_progress(states)
        else:
            states["layer2"] = ("skipped", "checkbox off")
            _render_progress(states)

        # ── Verdict banner ──
        # Allowlist wins over blacklist. If the URL is on the trusted
        # allowlist we never render the "Confirmed phishing" banner,
        # even if a feed has wrongly listed the domain.
        if bl_result["listed"] and not al_result["listed"]:
            match_human = (
                "Exact URL match" if bl_result["match_type"] == "exact_url"
                else "Hostname match"
            )
            st.markdown(
                f'<div class="verdict-confirmed">'
                f'🛑 Confirmed phishing'
                f'<div class="source-label">'
                f'Listed on <b>{bl_result["source"]}</b> · {match_human} · '
                f'verified by the threat-intelligence community'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            if label == "Phishing":
                st.markdown(
                    '<div class="verdict-phishing">'
                    '⚠ Phishing detected'
                    '<div class="source-label">Predicted by the URL feature model</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
            elif al_result["listed"]:
                st.markdown(
                    f'<div class="verdict-safe">'
                    f'✓ Trusted site'
                    f'<div class="source-label">'
                    f'Hostname matches the known-good allowlist '
                    f'(<b>{al_result["matched_domain"]}</b>)'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="verdict-safe">'
                    '✓ Looks safe'
                    '<div class="source-label">No flags from blacklist or URL feature model</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

        # ── AI ADVISORY ── (streams word-by-word like ChatGPT)
        st.markdown('<div class="section-title">Advisory</div>', unsafe_allow_html=True)
        advisory_text = build_advisory(
            url=url_clean,
            bl_result=bl_result,
            label=label,
            conf_phishing=conf_phishing,
            features=feats,
            layer2=layer2,
            al_result=al_result,
        )

        def _md_to_html(s: str) -> str:
            """Tiny markdown-to-HTML converter for the bits we use:
            **bold** and double-newline paragraph breaks."""
            import re
            s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
            s = s.replace("\n\n", "<br><br>")
            return s

        # The advisory streams into a single placeholder that we rewrite as each
        # word arrives. The wrapper div gives it the soft surface look from the mockup.
        advisory_box = st.empty()
        for partial in typewriter_stream(advisory_text, delay_per_word=0.025):
            advisory_box.markdown(
                f'<div style="background:#1e293b; border:0.5px solid #334155; '
                f'border-radius:10px; padding:18px 20px; font-size:14px; '
                f'line-height:1.7; color:#cbd5e1;">{_md_to_html(partial)}</div>',
                unsafe_allow_html=True,
            )


        st.markdown('<div class="section-title">Confidence Scores</div>', unsafe_allow_html=True)

        # ── Confidence scores ──
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Phishing", f"{conf_phishing*100:.1f}%")
            st.progress(conf_phishing)
        with col2:
            st.metric("Safe", f"{conf_safe*100:.1f}%")
            st.progress(conf_safe)
        with col3:
            if conf_phishing > 0.75:
                risk = "High"
            elif conf_phishing > 0.5:
                risk = "Medium"
            else:
                risk = "Low"
            st.metric("Risk Level", risk)

        # ═══════════════════════════════════════════════════════════
        # LAYER 2 — Page content analysis details
        # (Uses the layer2 result that was already computed in the progress flow.)
        # ═══════════════════════════════════════════════════════════
        if deep_scan and layer2 is not None:
            st.markdown('<div class="section-title">Deep Page Scan Details</div>', unsafe_allow_html=True)

            # Case 1: the page could not be fetched at all.
            if not layer2["success"]:
                st.warning(
                    f"Deep scan unavailable: {layer2['error']}. "
                    "Falling back to the Layer 1 verdict above."
                )
            else:
                # Case 2: page fetched successfully — show the 4 checks.
                score = layer2["suspicion_score"]
                if score >= 2:
                    st.error(f"Suspicion score: {score} / 4  ·  multiple red flags on the page")
                elif score == 1:
                    st.warning(f"Suspicion score: {score} / 4  ·  one indicator was raised")
                else:
                    st.success(f"Suspicion score: {score} / 4  ·  page passed all four checks")

                # Detail rows: tick or cross for each of the 4 checks.
                check_keys = [
                    "wallet_keywords",
                    "form_mismatch",
                    "redirect_chain",
                    "hidden_iframe",
                ]
                detail_rows = []
                for key in check_keys:
                    flagged = layer2[key] == 1
                    detail_rows.append({
                        "Check": CHECK_DESCRIPTIONS[key],
                        "Result": "Flagged" if flagged else "Clean",
                    })
                st.table(pd.DataFrame(detail_rows))

                # Extra info: where we actually ended up, and how many hops it took.
                st.caption(
                    f"Final URL after redirects: `{layer2['final_url']}`  "
                    f"·  Redirect hops: **{layer2['redirect_count']}**"
                )

        # ── Feature breakdown ──
        st.markdown('<div class="section-title">Feature Breakdown</div>', unsafe_allow_html=True)

        feat_df = pd.DataFrame([
            {
                "Feature":     k,
                "Value":       v,
                "Description": FEATURE_DESCRIPTIONS.get(k, ""),
            }
            for k, v in feats.items()
        ])

        col_feats, col_chart = st.columns([3, 2])
        with col_feats:
            st.dataframe(
                feat_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Feature":     st.column_config.TextColumn("Feature"),
                    "Value":       st.column_config.NumberColumn("Value", format="%.4f"),
                    "Description": st.column_config.TextColumn("Description"),
                }
            )
        with col_chart:
            # Top 10 features by importance
            if hasattr(model, "feature_importances_"):
                importance_map = dict(zip(feature_cols, model.feature_importances_))
                top10 = sorted(
                    [(k, importance_map.get(k, 0), feats.get(k, 0))
                     for k in feature_cols],
                    key=lambda x: x[1], reverse=True
                )[:10]

                top_names  = [t[0] for t in top10]
                top_vals   = [t[2] for t in top10]  # actual value for this URL
                top_imps   = [t[1] for t in top10]

                st.markdown("**Top 10 Most Important Features (this URL)**")
                for name, imp, val in top10:
                    label_disp = f"{name}: {val}"
                    bar_val = min(imp * 5, 1.0)
                    st.markdown(f"`{name}`  →  `{val}`")
                    st.progress(float(bar_val))

        # ── Suspicious signals callout ──
        st.markdown('<div class="section-title">⚠️ Suspicious Signals Detected</div>', unsafe_allow_html=True)
        signals = []
        if feats.get("has_ip_address"): signals.append("🔴 IP address used instead of domain name")
        if feats.get("brand_in_subdomain"): signals.append("🔴 Trusted brand name spoofed in subdomain")
        if feats.get("is_shortener"): signals.append("🟡 URL shortening service detected")
        if feats.get("suspicious_tld"): signals.append("🟡 Suspicious top-level domain (TLD)")
        if feats.get("at_count", 0) > 0: signals.append("🟡 '@' symbol present, browser redirect trick")
        if feats.get("double_slash"): signals.append("🟡 Double '//' in path, redirect obfuscation")
        if feats.get("has_hex_encoding"): signals.append("🟡 Excessive hex/percent encoding")
        if feats.get("keyword_count", 0) >= 2: signals.append(f"🟡 {feats['keyword_count']} suspicious keywords found (login, verify, paypal…)")
        if not feats.get("has_https"): signals.append("🟡 No HTTPS, connection is unencrypted")
        if feats.get("has_port"): signals.append("🟡 Non-standard port detected")
        if feats.get("domain_entropy", 0) > 3.8: signals.append("🟡 High domain entropy, domain looks randomly generated")

        if signals:
            for s in signals:
                st.markdown(f"- {s}")
        else:
            st.markdown("✅ No significant suspicious signals detected.")


# ═══════════════════════════════════════════════════════════
# PAGE 2 — BATCH ANALYSIS
# ═══════════════════════════════════════════════════════════
elif page == "Batch URL Analysis":
    st.markdown("# Batch Analysis")
    st.markdown(
        '<p style="color: #94a3b8; font-size: 0.95rem; margin-top: -8px; margin-bottom: 28px;">'
        'Upload a CSV of URLs to analyse them all at once. The file needs a column called <code>url</code>.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Upload a CSV file with a column named **`url`** (and optionally a **`label`** column). "
        "The system will classify every URL and let you download the results."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            df_in = pd.read_csv(uploaded)
            df_in.columns = [c.strip().lower() for c in df_in.columns]
            assert "url" in df_in.columns, "CSV must have a 'url' column"
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        st.markdown(f"Loaded **{len(df_in):,} URLs**. Running batch analysis…")
        progress = st.progress(0)
        results  = []
        for i, url in enumerate(df_in["url"].astype(str)):
            url_clean = url.strip()
            try:
                # Run the same three-layer pipeline used in single-URL mode.
                # Layer 0a (allowlist) and Layer 0b (blacklist) override the
                # model with a confidence floor of 0.99, matching the
                # priority order documented in Chapter 3.
                al_hit = allowlist_check(url_clean)
                bl_hit = blacklist.check(url_clean)
                raw_lbl, raw_cp, raw_cs, _ = predict_url(url_clean)
                if al_hit["listed"]:
                    lbl = "Safe"
                    cs  = max(raw_cs, 0.99)
                    cp  = 1 - cs
                    src = "Allowlist"
                elif bl_hit["listed"]:
                    lbl = "Phishing"
                    cp  = max(raw_cp, 0.99)
                    cs  = 1 - cp
                    src = f"Blacklist ({bl_hit['source']})"
                else:
                    lbl, cp, cs = raw_lbl, raw_cp, raw_cs
                    src = "ML Model"
            except Exception:
                lbl, cp, cs, src = "Error", 0.0, 0.0, "Error"
            results.append({"url": url, "prediction": lbl,
                             "confidence_phishing": round(cp, 4),
                             "confidence_safe": round(cs, 4),
                             "source": src})
            progress.progress((i + 1) / len(df_in))

        df_out = pd.DataFrame(results)
        if "label" in df_in.columns:
            df_out["true_label"] = df_in["label"].values

        st.success("Batch analysis complete!")

        # Summary stats
        c1, c2, c3 = st.columns(3)
        phish_count = (df_out["prediction"] == "Phishing").sum()
        safe_count  = (df_out["prediction"] == "Safe").sum()
        c1.metric("Total URLs", len(df_out))
        c2.metric("🚨 Phishing", phish_count)
        c3.metric("✅ Safe", safe_count)

        st.dataframe(df_out, use_container_width=True)

        # Download button
        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Results CSV",
            data=csv_bytes,
            file_name="phishguard_batch_results.csv",
            mime="text/csv",
        )


# ═══════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown("# Model Performance")
    st.markdown(
        '<p style="color: #94a3b8; font-size: 0.95rem; margin-top: -8px; margin-bottom: 28px;">'
        'Evaluation metrics, confusion matrix, ROC curve, and feature importance for '
        'the trained Random Forest model.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Trained on a labelled URL dataset using **Random Forest (300 estimators)** "
        "with 5-fold stratified cross-validation."
    )

    # Load report if exists
    report_path = os.path.join(MODEL_DIR, "model_report.txt")
    if os.path.exists(report_path):
        with open(report_path) as f:
            lines = f.readlines()
        metrics = {}
        for line in lines:
            if ":" in line:
                k, v = line.split(":", 1)
                try:
                    metrics[k.strip()] = float(v.strip())
                except ValueError:
                    pass

        if metrics:
            cols = st.columns(4)
            display = [
                ("accuracy",  "Accuracy"),
                ("precision", "Precision"),
                ("recall",    "Recall"),
                ("f1",        "F1 Score"),
            ]
            for col, (key, title) in zip(cols, display):
                val = metrics.get(key, 0)
                col.markdown(
                    f'<div class="metric-card"><div class="val">{val*100:.1f}%</div>'
                    f'<div class="lbl">{title}</div></div>',
                    unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)
            col5, col6 = st.columns(2)
            auc = metrics.get("auc", 0)
            cv  = metrics.get("cv_f1_mean", 0)
            cv_s= metrics.get("cv_f1_std", 0)
            col5.metric("ROC-AUC Score", f"{auc:.4f}")
            col6.metric("CV F1 (5-fold)", f"{cv:.4f} ± {cv_s:.4f}")

    # Plots
    st.markdown("---")
    img_col1, img_col2, img_col3 = st.columns(3)
    for col, fname, title in [
        (img_col1, "confusion_matrix.png",  "Confusion Matrix"),
        (img_col2, "roc_curve.png",          "ROC Curve"),
        (img_col3, "feature_importance.png", "Feature Importances"),
    ]:
        img_path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(img_path):
            col.image(img_path, caption=title, use_container_width=True)
        else:
            col.info(f"{title} not found. Run train_model.py to generate.")

    # Feature descriptions
    st.markdown("---")
    st.markdown("### 📐 Features Used (26 total)")
    st.markdown(
        "The model uses **26 lexical, structural, and content-based features** "
        "extracted purely from the URL string. No external DNS or WHOIS lookups required."
    )
    feat_desc_df = pd.DataFrame([
        {"Feature": k, "Description": v}
        for k, v in FEATURE_DESCRIPTIONS.items()
    ])
    st.dataframe(feat_desc_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════
# PAGE 4 — PREDICTION LOG
# ═══════════════════════════════════════════════════════════
elif page == "Prediction Log":
    st.markdown("# Prediction Log")
    st.markdown(
        '<p style="color: #94a3b8; font-size: 0.95rem; margin-top: -8px; margin-bottom: 28px;">'
        'Every URL analysed in this session, newest first.</p>',
        unsafe_allow_html=True,
    )
    st.markdown("Every URL analysed in this session is recorded below.")

    colA, colB = st.columns([5, 1])
    with colB:
        if st.button("🗑️ Clear Log"):
            clear_log()
            st.success("Log cleared.")

    rows = load_log()
    if rows:
        df_log = pd.DataFrame(rows)
        # Colour-code prediction column
        def _style_pred(val):
            if val == "Phishing":
                return "color: #ef4444; font-weight: bold"
            elif val == "Safe":
                return "color: #10b981; font-weight: bold"
            return ""
        # pandas >= 2.1 renamed applymap -> map; support both versions
        try:
            styled = df_log.style.map(_style_pred, subset=["prediction"])
        except AttributeError:
            styled = df_log.style.applymap(_style_pred, subset=["prediction"])
        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
        )
        # Download
        csv_bytes = df_log.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export Log CSV", csv_bytes,
                           "phishguard_log.csv", "text/csv")
    else:
        st.info("No predictions yet. Go to Single URL Analysis to get started.")


# ═══════════════════════════════════════════════════════════
# PAGE 5 — ABOUT
# ═══════════════════════════════════════════════════════════
elif page == "About":
    st.markdown("# About Klevr")
    st.markdown(
        '<p style="color: #94a3b8; font-size: 0.95rem; margin-top: -8px; margin-bottom: 28px;">'
        'A three-layer phishing URL detection system built as a final year project.</p>',
        unsafe_allow_html=True,
    )

    st.markdown("""
### Project Overview
**Klevr** is an academic final year project that develops a machine learning
system for detecting phishing URLs in real time using only lexical and structural
URL features. No external API calls or DNS lookups required.

### Research Problem
Phishing attacks remain one of the most prevalent cyber threats globally.
Attackers craft deceptive URLs that impersonate legitimate websites to steal
user credentials. Traditional blacklist-based defences are reactive and fail
against new domains. This system takes a proactive, AI-based approach.

### Methodology
1. **Dataset**: Labelled URL dataset (safe = 0, phishing = 1)
2. **Feature Engineering**: 26 handcrafted lexical, structural, and content-based features
3. **Model**: Random Forest Classifier (300 estimators, balanced class weights)
4. **Validation**: 80/20 train-test split with 5-fold stratified cross-validation
5. **Metrics**: Accuracy, Precision, Recall, F1 Score, ROC-AUC

### Feature Categories
| Category | Count | Examples |
|---|---|---|
| Lexical / Length | 5 | URL length, domain length, path length |
| Character Composition | 11 | Dot count, digit ratio, Shannon entropy |
| Structural / Syntax | 3 | Subdomain count, port detection |
| Protocol / Obfuscation | 4 | HTTPS, IP in URL, shorteners, hex encoding |
| Content / Keyword | 3 | Suspicious keywords, brand spoofing, TLD |

### Technology Stack
- **Language**: Python 3
- **ML Framework**: Scikit-learn (Random Forest)
- **Data Processing**: Pandas, NumPy
- **URL Parsing**: tldextract, urllib
- **Web Interface**: Streamlit
- **Visualisation**: Matplotlib, Seaborn

---
**Student**: Olubo Demilade  |  **Matric No**: 22/11227  
**Department**: Computer Science, Caleb University, Imota, Lagos  
**Session**: 2025/2026  
**Supervisor**: Prof. M. K. Aregbesola
""")