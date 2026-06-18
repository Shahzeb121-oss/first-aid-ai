import streamlit as st
from src.predict import predict
from src.utils import get_severity_icon, get_severity_meta

st.set_page_config(
    page_title="MediAssist AI — First Aid Emergency Assistant",
    page_icon="🚑",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 780px; }

/* ── Page background ── */
.stApp { background: #0d1117; }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
    border-bottom: 1px solid #21262d;
    margin-bottom: 2.5rem;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(248, 81, 73, 0.12);
    border: 1px solid rgba(248, 81, 73, 0.3);
    color: #f85149;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 1.25rem;
}
.hero-badge::before {
    content: '';
    width: 6px; height: 6px;
    background: #f85149;
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.8); }
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 600;
    color: #e6edf3;
    letter-spacing: -0.03em;
    line-height: 1.2;
    margin: 0 0 0.75rem;
}
.hero h1 span { color: #f85149; }
.hero p {
    font-size: 1rem;
    color: #8b949e;
    margin: 0;
    line-height: 1.6;
}

/* ── Search box ── */
.stTextInput > div > div > input {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 12px !important;
    color: #e6edf3 !important;
    font-size: 1rem !important;
    padding: 0.9rem 1.25rem !important;
    height: auto !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
.stTextInput > div > div > input:focus {
    border-color: #f85149 !important;
    box-shadow: 0 0 0 3px rgba(248,81,73,0.15) !important;
}
.stTextInput > div > div > input::placeholder { color: #484f58 !important; }
.stTextInput label { color: #8b949e !important; font-size: 0.85rem !important; font-weight: 500 !important; letter-spacing: 0.03em; }

/* ── Analyze button ── */
.stButton > button {
    width: 100%;
    background: #f85149 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.5rem !important;
    letter-spacing: 0.02em;
    transition: all 0.2s ease !important;
    margin-top: 0.5rem !important;
}
.stButton > button:hover {
    background: #da3633 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(248,81,73,0.35) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Cards ── */
.result-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s ease;
}
.result-card:hover { border-color: #30363d; }

.card-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #484f58;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 6px;
}
.card-label::before {
    content: '';
    width: 3px; height: 3px;
    background: #f85149;
    border-radius: 50%;
}

.situation-name {
    font-size: 1.5rem;
    font-weight: 600;
    color: #e6edf3;
    letter-spacing: -0.02em;
    line-height: 1.3;
}

/* ── Severity badges ── */
.sev-critical {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(248,81,73,0.12);
    border: 1px solid rgba(248,81,73,0.35);
    color: #f85149;
    padding: 8px 16px; border-radius: 8px;
    font-weight: 600; font-size: 0.9rem;
}
.sev-severe {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(210,153,34,0.12);
    border: 1px solid rgba(210,153,34,0.35);
    color: #d29922;
    padding: 8px 16px; border-radius: 8px;
    font-weight: 600; font-size: 0.9rem;
}
.sev-moderate {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(63,185,80,0.1);
    border: 1px solid rgba(63,185,80,0.3);
    color: #3fb950;
    padding: 8px 16px; border-radius: 8px;
    font-weight: 600; font-size: 0.9rem;
}

/* ── Steps ── */
.steps-list { list-style: none; padding: 0; margin: 0; }
.step-item {
    display: flex; align-items: flex-start; gap: 14px;
    padding: 0.75rem 0;
    border-bottom: 1px solid #21262d;
}
.step-item:last-child { border-bottom: none; }
.step-num {
    flex-shrink: 0;
    width: 26px; height: 26px;
    background: rgba(248,81,73,0.12);
    border: 1px solid rgba(248,81,73,0.25);
    border-radius: 6px;
    color: #f85149;
    font-size: 11px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    display: flex; align-items: center; justify-content: center;
}
.step-text { color: #c9d1d9; font-size: 0.9rem; line-height: 1.6; padding-top: 3px; }

/* ── Info panels ── */
.info-panel {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1rem 1.25rem;
}
.info-panel-title {
    font-size: 11px; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: #484f58; margin-bottom: 0.6rem;
}
.info-panel-body { color: #8b949e; font-size: 0.875rem; line-height: 1.7; }

/* ── Warning panel ── */
.warning-panel {
    background: rgba(210,153,34,0.06);
    border: 1px solid rgba(210,153,34,0.2);
    border-radius: 10px;
    padding: 1rem 1.25rem;
}
.warning-panel-title {
    font-size: 11px; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: #d29922; margin-bottom: 0.6rem;
}
.warning-panel-body { color: #c9d1d9; font-size: 0.875rem; line-height: 1.7; }

/* ── Emergency trigger panel ── */
.trigger-panel {
    background: rgba(248,81,73,0.06);
    border: 1px solid rgba(248,81,73,0.2);
    border-left: 3px solid #f85149;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.25rem;
}
.trigger-panel-title {
    font-size: 11px; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: #f85149; margin-bottom: 0.6rem;
}
.trigger-panel-body { color: #c9d1d9; font-size: 0.875rem; line-height: 1.7; }

/* ── Confidence bar ── */
.conf-bar-wrap {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 0.9rem 1.25rem;
    display: flex; align-items: center; gap: 16px;
}
.conf-label { font-size: 11px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #484f58; white-space: nowrap; }
.conf-track { flex: 1; height: 4px; background: #21262d; border-radius: 2px; overflow: hidden; }
.conf-fill-high { height: 100%; background: #3fb950; border-radius: 2px; transition: width 0.6s ease; }
.conf-fill-mid  { height: 100%; background: #d29922; border-radius: 2px; transition: width 0.6s ease; }
.conf-fill-low  { height: 100%; background: #f85149; border-radius: 2px; transition: width 0.6s ease; }
.conf-pct { font-size: 13px; font-weight: 600; font-family: 'JetBrains Mono', monospace; white-space: nowrap; }

/* ── Quick examples ── */
.examples-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 1rem; }
.example-chip {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.8rem;
    color: #8b949e;
    cursor: pointer;
    transition: all 0.15s ease;
}
.example-chip:hover { border-color: #f85149; color: #f85149; }

/* ── Divider ── */
.section-divider {
    border: none; border-top: 1px solid #21262d;
    margin: 1.5rem 0;
}

/* ── Emergency banner ── */
.emergency-banner {
    background: rgba(248,81,73,0.1);
    border: 1px solid rgba(248,81,73,0.3);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 1.5rem;
}
.emergency-banner-icon { font-size: 1.4rem; flex-shrink: 0; }
.emergency-banner-text { color: #f85149; font-weight: 600; font-size: 0.95rem; line-height: 1.4; }
.emergency-banner-sub { color: #c9d1d9; font-weight: 400; font-size: 0.85rem; margin-top: 2px; }

/* ── Footer ── */
.page-footer {
    text-align: center;
    color: #484f58;
    font-size: 0.78rem;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid #21262d;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">Live Emergency Guidance</div>
    <h1>First Aid <span>Emergency</span> Assistant</h1>
    <p>Describe any medical emergency in plain language.<br>Get instant, structured first aid guidance.</p>
</div>
""", unsafe_allow_html=True)

# ── Search ────────────────────────────────────────────────────────────────────
query = st.text_input(
    "DESCRIBE THE EMERGENCY",
    placeholder="e.g. my kid burned their hand on the stove, someone is choking, dog bite on arm...",
    key="query_input",
)

col_btn, _ = st.columns([1, 2])
with col_btn:
    search = st.button("⚡  Analyse Emergency", use_container_width=True)

# Quick example chips (visual only — for guidance)
st.markdown("""
<div style="margin-top:0.75rem;">
    <span style="font-size:11px;color:#484f58;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;">Try:</span>
    <span style="font-size:0.8rem;color:#484f58;margin-left:8px;">
        "chest pain and sweating" &nbsp;·&nbsp; "baby choking" &nbsp;·&nbsp; "severe burn" &nbsp;·&nbsp; "snake bite" &nbsp;·&nbsp; "can't breathe"
    </span>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ── Results ───────────────────────────────────────────────────────────────────
if search:
    if not query.strip():
        st.markdown("""
        <div style="background:rgba(210,153,34,0.08);border:1px solid rgba(210,153,34,0.25);border-radius:10px;padding:1rem 1.25rem;color:#d29922;font-size:0.9rem;">
            Please describe the emergency before analysing.
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner(""):
            result, score = predict(query)

        if "Error" in result:
            st.markdown(f"""
            <div style="background:rgba(248,81,73,0.08);border:1px solid rgba(248,81,73,0.25);border-radius:10px;padding:1rem 1.25rem;color:#f85149;font-size:0.9rem;">
                {result['Error']}
            </div>
            """, unsafe_allow_html=True)
        else:
            clean = {str(k).strip().lower(): v for k, v in result.items()}

            situation = clean.get("situation",         "Unknown Emergency")
            severity  = clean.get("severity",          "Moderate")
            steps_raw = clean.get("firstaidsteps",     "No specific steps found.")
            medicines = clean.get("medicines",         "None specified.")
            triggers  = clean.get("emergencytriggers", "None specified.")
            warnings  = clean.get("warnings",          "Seek professional care if condition worsens.")

            sev_lower = severity.strip().lower()
            is_critical = sev_lower in ("life-threatening", "critical", "severe")

            # ── Emergency banner for critical cases ──
            if is_critical:
                st.markdown("""
                <div class="emergency-banner">
                    <div class="emergency-banner-icon">🚨</div>
                    <div>
                        <div class="emergency-banner-text">Call emergency services immediately</div>
                        <div class="emergency-banner-sub">
    🚑 Emergency Ambulance: 1122 (Sindh,Pakistan)
</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Situation + Severity row ──
            sev_css = "sev-critical" if sev_lower in ("life-threatening","critical","severe") else ("sev-severe" if "moderate to severe" in sev_lower else "sev-moderate")
            sev_dot = "🔴" if "critical" in sev_lower or "life" in sev_lower or sev_lower == "severe" else ("🟡" if "moderate" in sev_lower else "🟢")

            st.markdown(f"""
            <div class="result-card">
                <div class="card-label">Identified situation</div>
                <div class="situation-name">{situation}</div>
                <div style="margin-top:1rem;">
                    <span class="{sev_css}">{sev_dot} {severity}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── First Aid Steps ──
            steps = [s.strip() for s in str(steps_raw).split(";") if s.strip()]
            steps_html = ""
            for i, step in enumerate(steps, 1):
                clean_step = step.lstrip("0123456789.) ")
                if clean_step:
                    steps_html += f"""
                    <li class="step-item">
                        <div class="step-num">{i:02d}</div>
                        <div class="step-text">{clean_step}</div>
                    </li>"""

            st.markdown(f"""
            <div class="result-card">
                <div class="card-label">First aid steps</div>
                <ul class="steps-list">{steps_html}</ul>
            </div>
            """, unsafe_allow_html=True)

            # ── Medicines + Triggers (two columns) ──
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="info-panel">
                    <div class="info-panel-title">💊 Recommended supplies</div>
                    <div class="info-panel-body">{medicines}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="trigger-panel">
                    <div class="trigger-panel-title">🚨 Emergency triggers</div>
                    <div class="trigger-panel-body">{triggers}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

            # ── Warnings ──
            st.markdown(f"""
            <div class="warning-panel">
                <div class="warning-panel-title">⚠️ Critical warnings</div>
                <div class="warning-panel-body">{warnings}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

            # ── Confidence bar ──
            pct = round(score * 100, 1)
            fill_class = "conf-fill-high" if pct >= 60 else ("conf-fill-mid" if pct >= 35 else "conf-fill-low")
            conf_color = "#3fb950" if pct >= 60 else ("#d29922" if pct >= 35 else "#f85149")
            conf_label = "High confidence" if pct >= 60 else ("Moderate confidence" if pct >= 35 else "Low — verify manually")

            st.markdown(f"""
            <div class="conf-bar-wrap">
                <span class="conf-label">Match confidence</span>
                <div class="conf-track">
                    <div class="{fill_class}" style="width:{min(pct,100)}%;"></div>
                </div>
                <span class="conf-pct" style="color:{conf_color};">{pct}%</span>
                <span style="font-size:11px;color:#484f58;">{conf_label}</span>
            </div>
            """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-footer">
    <strong style="color:#8b949e;">⚕ Medical Disclaimer</strong><br>
    This tool provides first aid reference information only. It is not a substitute for professional medical advice.<br>
    In any life-threatening situation, call emergency services immediately.
</div>
""", unsafe_allow_html=True)
