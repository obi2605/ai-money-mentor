# ==============================================================================
# app.py
# AI Money Mentor — Main Streamlit Application
# Run with: streamlit run app.py
# ------------------------------------------------------------------------------
# ARCHITECTURE:
#   This file owns: UI rendering, session state, routing decisions, and
#   orchestrating calls between llm_orchestrator.py and quant_engine.py.
#   It is intentionally the ONLY file with Streamlit imports.
# ==============================================================================

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import asdict
from typing import Optional

import plotly.graph_objects as go
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Internal modules
import quant_engine as qe
from llm_orchestrator import (
    Intent,
    detect_intent,
    extract_fire_params,
    extract_health_params,
    extract_market_params,
    extract_sip_params,
    format_history,
    generate_clarification_request,
    generate_general_response,
    generate_response,
)

load_dotenv()
logger = logging.getLogger(__name__)

# ============================================================================ #
#  SECTION 1 — PAGE CONFIG & ET BRANDING                                       #
# ============================================================================ #

st.set_page_config(
    page_title="AI Money Mentor | Economic Times",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------- #
#  ET Brand Palette & Custom CSS                                                #
#  Colors: ET Red #E2001A | Off-Black #1A1A1A | White #FFFFFF | Grey #F5F5F5  #
# ---------------------------------------------------------------------------- #
ET_CSS = """
<style>
/* ── Google Font Import ────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Source+Sans+3:wght@400;600&display=swap');

/* ── Root Variables ─────────────────────────────────────────────────────────── */
:root {
    --et-red:       #E2001A;
    --et-red-dark:  #B5001A;
    --et-black:     #1A1A1A;
    --et-white:     #FFFFFF;
    --et-grey:      #F5F5F5;
    --et-grey-mid:  #D0D0D0;
    --et-text:      #2C2C2C;
    --et-subtext:   #5A5A5A;
    --font-display: 'Playfair Display', Georgia, serif;
    --font-body:    'Source Sans 3', sans-serif;
    --radius:       8px;
    --shadow:       0 2px 12px rgba(0,0,0,0.08);
}

/* ── Global Reset ────────────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    font-family: var(--font-body);
    background-color: var(--et-grey);
    color: var(--et-text);
}

/* ── Header Bar ──────────────────────────────────────────────────────────────── */
[data-testid="stHeader"] { background-color: var(--et-black); }
.et-topbar {
    background: var(--et-black);
    color: var(--et-white);
    padding: 10px 24px;
    display: flex;
    align-items: center;
    gap: 12px;
    border-bottom: 3px solid var(--et-red);
    margin-bottom: 0;
}
.et-topbar .brand-logo {
    font-family: var(--font-display);
    font-size: 22px;
    font-weight: 900;
    color: var(--et-red);
    letter-spacing: -0.5px;
}
.et-topbar .brand-label {
    font-size: 13px;
    color: var(--et-grey-mid);
    letter-spacing: 0.8px;
    text-transform: uppercase;
}
.et-topbar .divider { color: #444; margin: 0 8px; }

/* ── Sidebar ─────────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--et-black);
    border-right: 1px solid #333;
}
[data-testid="stSidebar"] * { color: var(--et-white) !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stFileUploader label { color: var(--et-grey-mid) !important; }

/* ── Chat Messages ────────────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: var(--radius);
    margin-bottom: 4px;
    border: none;
    box-shadow: var(--shadow);
}

/* Force text visible in ALL chat bubbles */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] label {
    color: var(--et-text) !important;
}

/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: var(--et-white);
}

/* Assistant bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: var(--et-white);
    border-left: 3px solid var(--et-red);
}

/* Remove the blank top padding block Streamlit injects before message content */
[data-testid="stChatMessage"] > div:first-child {
    padding-top: 0 !important;
    margin-top: 0 !important;
}
[data-testid="stChatMessage"] [data-testid="stVerticalBlock"] > div:empty {
    display: none !important;
}

/* ── Metric Cards ─────────────────────────────────────────────────────────────── */
.metric-card {
    background: var(--et-white);
    border-radius: var(--radius);
    padding: 16px 20px;
    box-shadow: var(--shadow);
    border-top: 3px solid var(--et-red);
    text-align: center;
}
.metric-card .metric-value {
    font-family: var(--font-display);
    font-size: 28px;
    font-weight: 700;
    color: var(--et-black);
}
.metric-card .metric-label {
    font-size: 12px;
    color: var(--et-subtext);
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-top: 4px;
}

/* ── Score Badge ──────────────────────────────────────────────────────────────── */
.score-badge {
    display: inline-block;
    font-family: var(--font-display);
    font-size: 48px;
    font-weight: 900;
    padding: 12px 28px;
    border-radius: var(--radius);
    color: white;
}
.score-a-plus { background: #1B9E4E; }
.score-a      { background: #2ECC71; }
.score-b      { background: #F39C12; }
.score-c      { background: #E67E22; }
.score-d      { background: var(--et-red); }

/* ── Chat Input ────────────────────────────────────────────────────────────────── */
[data-testid="stChatInput"] textarea {
    color: #FFFFFF !important;
    caret-color: #FFFFFF !important;
    font-family: var(--font-body) !important;
    font-size: 14px !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: rgba(255,255,255,0.5) !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────────────── */
.stButton button {
    background: var(--et-red) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    transition: background 0.2s;
}
.stButton button:hover { background: var(--et-red-dark) !important; }

/* ── Section Headings ────────────────────────────────────────────────────────── */
.section-heading {
    font-family: var(--font-display);
    font-size: 20px;
    font-weight: 700;
    color: var(--et-black);
    border-bottom: 2px solid var(--et-red);
    padding-bottom: 6px;
    margin-bottom: 16px;
}

/* ── Disclaimer ──────────────────────────────────────────────────────────────── */
.disclaimer {
    font-size: 10px;
    color: var(--et-subtext);
    text-align: center;
    padding: 8px;
    border-top: 1px solid var(--et-grey-mid);
    margin-top: 16px;
}

/* ── Quick-Start Chips ────────────────────────────────────────────────────────── */
.chip-row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
</style>
"""
st.markdown(ET_CSS, unsafe_allow_html=True)

# ============================================================================ #
#  SECTION 2 — SESSION STATE INITIALISATION                                    #
# ============================================================================ #

def _init_state() -> None:
    """Initialise all session state keys with safe defaults (idempotent)."""
    defaults: dict = {
        "messages": [],           # list[dict{role, content}]
        "active_module": None,
        "user_context": {},
        "cams_data": None,
        "fire_result": None,
        "health_result": None,
        "sip_result": None,
        "market_result": None,
        "processing": False,
        "xray_report": None,
        "msg_results": {},        # Per-message results: {msg_idx: {"type": ..., "result": ...}}
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()

# ============================================================================ #
#  SECTION 3 — SIDEBAR                                                         #
# ============================================================================ #

with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 16px 0;">
        <div style="font-family: 'Playfair Display', serif; font-size: 20px;
                    font-weight: 900; color: #E2001A; letter-spacing: -0.5px;">
            ET Money Mentor
        </div>
        <div style="font-size: 11px; color: #888; letter-spacing: 0.8px;
                    text-transform: uppercase; margin-top: 2px;">
            AI Financial Advisor
        </div>
    </div>
    <hr style="border-color: #333; margin: 0 0 16px 0;">
    """, unsafe_allow_html=True)

    # --- Module Selector ---
    st.markdown("**📊 Choose a Tool**")
    module_options = {
        "💬 Chat Freely": None,
        "🏥 Money Health Score": Intent.HEALTH_SCORE,
        "🔥 FIRE Path Planner": Intent.FIRE_PLANNER,
        "📈 SIP Projector": Intent.SIP_PROJECTION,
        "📉 Market Data": Intent.MARKET_DATA,
        "🔬 MF Portfolio X-Ray": Intent.MF_XRAY,
    }
    selected_module_label = st.selectbox(
        "Module", list(module_options.keys()), label_visibility="collapsed"
    )
    forced_intent: Optional[Intent] = module_options[selected_module_label]

    st.markdown("<hr style='border-color:#333;margin:12px 0;'>", unsafe_allow_html=True)

    # --- CAMS PDF Uploader ---
    st.markdown("**📄 Upload CAMS Statement**")
    st.caption("Processed 100% locally. Never sent to any API.")
    uploaded_file = st.file_uploader(
        "CAMS PDF", type=["pdf"], label_visibility="collapsed"
    )
    if uploaded_file and st.session_state.cams_data is None:
        with st.spinner("Parsing locally..."):
            try:
                # Save to temp file and parse (privacy_parser.py handles this)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                # Import lazily to avoid circular imports at module load
                from privacy_parser import parse_cams_pdf
                st.session_state.cams_data = parse_cams_pdf(tmp_path)
                os.unlink(tmp_path)  # Delete temp file immediately after parsing
                st.success(
                    f"✅ Parsed {st.session_state.cams_data.get('num_transactions', 0)} transactions"
                )
            except ImportError:
                st.warning("privacy_parser.py not yet built. Coming in Sprint 4.")
            except Exception as e:
                st.error(f"Parse error: {e}")
                logger.error("CAMS parse error: %s", e)

    st.markdown("<hr style='border-color:#333;margin:12px 0;'>", unsafe_allow_html=True)

    # --- Clear Chat ---
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.user_context = {}
        st.session_state.msg_results = {}
        st.session_state.fire_result = None
        st.session_state.health_result = None
        st.session_state.sip_result = None
        st.session_state.market_result = None
        st.session_state.cams_data = None
        st.session_state.xray_report = None
        st.rerun()

    # --- Disclaimer ---
    st.markdown("""
    <div class="disclaimer" style="color: #666;">
        For educational purposes only.<br>Not SEBI-registered financial advice.
    </div>
    """, unsafe_allow_html=True)

# ============================================================================ #
#  SECTION 4 — TOP HEADER BAR                                                  #
# ============================================================================ #

st.markdown("""
<div class="et-topbar">
    <span class="brand-logo">ET</span>
    <span class="divider">|</span>
    <span class="brand-label">AI Money Mentor &nbsp;·&nbsp; Powered by Economic Times</span>
</div>
""", unsafe_allow_html=True)

# ============================================================================ #
#  SECTION 5 — VISUALISATION HELPERS                                           #
# ============================================================================ #

def render_health_score(health: dict, key_suffix: str = "0") -> None:
    """Render the 6-dimension health score as a radar chart + metric cards."""
    dims = health["dimensions"]
    composite = health["composite_score"]
    grade = health["grade"]

    # Grade → CSS class mapping
    grade_class = {
        "A+": "score-a-plus", "A": "score-a",
        "B": "score-b", "C": "score-c", "D": "score-d"
    }.get(grade[0], "score-d")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"""
        <div style="text-align:center; padding: 20px;">
            <div style="font-size:13px; color:#5A5A5A; margin-bottom:8px;
                        text-transform:uppercase; letter-spacing:0.6px;">
                Money Health Score
            </div>
            <div class="score-badge {grade_class}">{composite:.0f}</div>
            <div style="margin-top:10px; font-family:'Playfair Display',serif;
                        font-size:18px; font-weight:700;">{grade}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Radar chart
        categories = [
            "Emergency Fund", "Insurance", "Diversification",
            "Debt Health", "Tax Efficiency", "Retirement"
        ]
        values_list = [
            dims["emergency_preparedness"], dims["insurance_coverage"],
            dims["investment_diversity"], dims["debt_health"],
            dims["tax_efficiency"], dims["retirement_readiness"],
        ]
        # Close the radar loop
        categories_loop = categories + [categories[0]]
        values_loop = values_list + [values_list[0]]

        fig = go.Figure(go.Scatterpolar(
            r=values_loop,
            theta=categories_loop,
            fill="toself",
            fillcolor="rgba(226,0,26,0.15)",
            line=dict(color="#E2001A", width=2),
            marker=dict(color="#E2001A", size=6),
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True, range=[0, 100],
                    tickfont=dict(size=9, color="#888"),
                    gridcolor="#E0E0E0",
                ),
                angularaxis=dict(tickfont=dict(size=10, color="#1A1A1A")),
                bgcolor="white",
            ),
            paper_bgcolor="white",
            margin=dict(t=20, b=20, l=20, r=20),
            height=280,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"chart_health_radar_{key_suffix}")
    st.markdown('<div class="section-heading">Dimension Breakdown</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    dim_labels = {
        "emergency_preparedness": "🛡️ Emergency Fund",
        "insurance_coverage": "📋 Insurance",
        "investment_diversity": "🎯 Diversification",
        "debt_health": "💳 Debt Health",
        "tax_efficiency": "📊 Tax Efficiency",
        "retirement_readiness": "🎯 Retirement",
    }
    for i, (key, label) in enumerate(dim_labels.items()):
        with cols[i % 3]:
            score_val = dims[key]
            color = "#1B9E4E" if score_val >= 70 else ("#F39C12" if score_val >= 40 else "#E2001A")
            st.markdown(f"""
            <div class="metric-card" style="border-top-color:{color};">
                <div class="metric-value" style="color:{color};">{score_val:.0f}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)


def render_sip_chart(sip_result: qe.SIPProjectionResult, key_suffix: str = "0") -> None:
    """Render an area chart of SIP corpus growth over time."""
    df = sip_result.monthly_schedule

    # Sample to yearly data points for a clean chart
    yearly_df = df[df["month"] % 12 == 0].copy()
    yearly_df["year_label"] = yearly_df["year"].astype(str) + "Y"

    # total_invested may be absent in FIRE roadmap schedules built before this fix
    if "total_invested" not in yearly_df.columns:
        yearly_df["total_invested"] = yearly_df["monthly_sip"].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yearly_df["year_label"],
        y=yearly_df["corpus_value"],
        name="Corpus Value",
        fill="tozeroy",
        fillcolor="rgba(226,0,26,0.12)",
        line=dict(color="#E2001A", width=2.5),
        mode="lines+markers",
        marker=dict(size=7, color="#E2001A"),
    ))
    fig.add_trace(go.Scatter(
        x=yearly_df["year_label"],
        y=yearly_df["total_invested"],
        name="Total Invested",
        line=dict(color="#1A1A1A", width=1.5, dash="dash"),
        mode="lines",
    ))

    if sip_result.target_corpus > 0:
        fig.add_hline(
            y=sip_result.target_corpus,
            line_dash="dot", line_color="#F39C12",
            annotation_text="Target",
            annotation_position="bottom right",
        )

    fig.update_layout(
        title=dict(
            text=f"SIP Growth Projection — {sip_result.years} Years @ {sip_result.assumed_cagr_pct}% CAGR",
            font=dict(family="'Playfair Display', serif", size=16, color="#1A1A1A"),
        ),
        xaxis=dict(title="Year", gridcolor="#F0F0F0"),
        yaxis=dict(
            title="Value (₹)",
            gridcolor="#F0F0F0",
            tickformat=",.0f",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(t=50, b=40, l=60, r=20),
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True, key=f"chart_sip_growth_{key_suffix}")

    # Summary metrics
    cols = st.columns(3)
    corpus_fmt = f"₹{sip_result.projected_corpus/1e5:.1f}L" if sip_result.projected_corpus < 1e7 \
        else f"₹{sip_result.projected_corpus/1e7:.2f}Cr"
    invested_fmt = f"₹{sip_result.monthly_schedule['total_invested'].iloc[-1]/1e5:.1f}L"
    gains = sip_result.projected_corpus - sip_result.monthly_schedule["total_invested"].iloc[-1]
    gains_fmt = f"₹{gains/1e5:.1f}L" if gains < 1e7 else f"₹{gains/1e7:.2f}Cr"

    cards = [
        ("Projected Corpus", corpus_fmt, "#1B9E4E"),
        ("Total Invested", invested_fmt, "#1A1A1A"),
        ("Total Gains", gains_fmt, "#E2001A"),
    ]
    for col, (label, val, color) in zip(cols, cards):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-top-color:{color};">
                <div class="metric-value" style="color:{color};">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)


def render_fire_roadmap(roadmap, key_suffix: str = "0") -> None:
    """Render the full FIRE roadmap: corpus chart, stress test, Monte Carlo fan."""
    df = roadmap.monthly_schedule
    yearly = df[df["month"] % 12 == 0].copy()

    def _fmt(v):
        if abs(v) >= 1e7: return f"₹{v/1e7:.2f}Cr"
        return f"₹{abs(v)/1e5:.1f}L"

    tab1, tab2, tab3 = st.tabs(["📈 Roadmap", "🔥 Stress Test", "🎲 Monte Carlo"])

    with tab1:
        on_track = roadmap.shortfall_surplus >= 0
        c1, c2, c3, c4 = st.columns(4)
        surplus_label = "Surplus" if on_track else "Shortfall"
        surplus_color = "#1B9E4E" if on_track else "#E2001A"
        for col, (label, val, color) in zip([c1, c2, c3, c4], [
            ("Target Corpus", _fmt(roadmap.required_corpus), "#1A1A1A"),
            ("Projected Corpus", _fmt(roadmap.projected_corpus), surplus_color),
            (surplus_label, _fmt(abs(roadmap.shortfall_surplus)), surplus_color),
            ("Required SIP", f"₹{roadmap.required_monthly_sip:,.0f}/mo",
             "#1B9E4E" if on_track else "#E2001A"),
        ]):
            col.markdown(f"""
            <div class="metric-card" style="border-top-color:{color};">
                <div class="metric-value" style="color:{color};">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

        # Show user's current SIP vs required SIP as a clear callout
        user_sip = roadmap.user_monthly_sip
        req_sip = roadmap.required_monthly_sip
        gap = req_sip - user_sip
        if gap > 0:
            st.warning(
                f"📊 Your current SIP of **₹{user_sip:,.0f}/mo** projects to "
                f"**{_fmt(roadmap.projected_corpus)}** — a shortfall of **{_fmt(abs(roadmap.shortfall_surplus))}**. "
                f"Increase to **₹{req_sip:,.0f}/mo** (↑₹{gap:,.0f}) to stay on track."
            )
        else:
            st.success(
                f"✅ Your current SIP of **₹{user_sip:,.0f}/mo** is sufficient. "
                f"You're projected to reach **{_fmt(roadmap.projected_corpus)}** vs target **{_fmt(roadmap.required_corpus)}**."
            )

        st.markdown("<br>", unsafe_allow_html=True)

        CHART_FONT = dict(color="#1A1A1A", family="'Source Sans 3', sans-serif", size=11)

        # Corpus growth curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=yearly["age"], y=yearly["corpus_value"], name="Projected Corpus",
            fill="tozeroy", fillcolor="rgba(226,0,26,0.10)",
            line=dict(color="#E2001A", width=2.5), mode="lines+markers",
            marker=dict(size=5), hovertemplate="Age %{x}: ₹%{y:,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=yearly["age"], y=yearly["target_corpus"], name="Target Corpus",
            line=dict(color="#1A1A1A", width=1.5, dash="dash"), mode="lines",
        ))
        # Annotate glide path shifts — only show every 3rd change to avoid overlap
        prev_alloc = None
        annotation_count = 0
        for m in roadmap.milestones:
            key = (m.allocation.equity_pct, m.allocation.debt_pct)
            if key != prev_alloc:
                if annotation_count % 3 == 0:  # Show every 3rd transition only
                    fig.add_annotation(
                        x=m.age, y=m.projected_corpus,
                        text=f"E:{m.allocation.equity_pct:.0f}% D:{m.allocation.debt_pct:.0f}%",
                        showarrow=True, arrowhead=2, arrowcolor="#AAA",
                        font=dict(size=9, color="#333"), bgcolor="white",
                        bordercolor="#DDD", borderwidth=1,
                        ax=0, ay=-36, standoff=6,
                    )
                prev_alloc = key
                annotation_count += 1

        fig.update_layout(
            title=dict(text=f"FIRE Roadmap — Age {roadmap.current_age} → {roadmap.retirement_age}",
                       font=dict(family="'Playfair Display', serif", size=17, color="#1A1A1A")),
            xaxis=dict(title="Age", gridcolor="#F0F0F0", dtick=5,
                       tickfont=CHART_FONT, title_font=CHART_FONT),
            yaxis=dict(title="Corpus (₹)", gridcolor="#F0F0F0", tickformat=",.0f",
                       tickfont=CHART_FONT, title_font=CHART_FONT),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        font=CHART_FONT),
            font=CHART_FONT,
            paper_bgcolor="white", plot_bgcolor="white",
            margin=dict(t=55, b=40, l=70, r=20), height=380,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"chart_fire_corpus_{key_suffix}")

        # Glide path stacked bar
        if roadmap.milestones:
            alloc_df = pd.DataFrame([{
                "Age": m.age, "Equity": m.allocation.equity_pct,
                "Debt": m.allocation.debt_pct, "Gold": m.allocation.gold_pct,
            } for m in roadmap.milestones[::2]])
            fig2 = go.Figure()
            for name, color in [("Equity", "#E2001A"), ("Debt", "#1A1A1A"), ("Gold", "#F0A500")]:
                fig2.add_trace(go.Bar(x=alloc_df["Age"], y=alloc_df[name],
                                      name=name, marker_color=color))
            fig2.update_layout(
                barmode="stack",
                title=dict(text="Asset Allocation Glide Path",
                           font=dict(family="'Playfair Display',serif", size=14, color="#1A1A1A"),
                           x=0, pad=dict(b=10)),
                xaxis=dict(title="Age", gridcolor="#F0F0F0",
                           tickfont=CHART_FONT, title_font=CHART_FONT),
                yaxis=dict(title="%", range=[0, 100], gridcolor="#F0F0F0",
                           tickfont=CHART_FONT, title_font=CHART_FONT),
                font=CHART_FONT,
                paper_bgcolor="white", plot_bgcolor="white",
                margin=dict(t=40, b=60, l=50, r=20), height=260,
                legend=dict(
                    orientation="h", y=-0.25, x=0,
                    font=CHART_FONT, bgcolor="rgba(0,0,0,0)",
                ),
            )
            st.plotly_chart(fig2, use_container_width=True, key=f"chart_fire_glide_{key_suffix}")

        # Backtest sub-panel
        if roadmap.backtest_result:
            bt = roadmap.backtest_result
            st.markdown('<div class="section-heading">📊 Historical Backtest (Real Nifty 50 Data)</div>',
                        unsafe_allow_html=True)
            b1, b2, b3, b4 = st.columns(4)
            for col, (label, val, color) in zip([b1, b2, b3, b4], [
                ("Actual XIRR", f"{bt.actual_xirr_pct:.1f}%", "#1B9E4E"),
                ("Nifty 50 CAGR", f"{bt.benchmark_cagr_pct:.1f}%", "#1A1A1A"),
                ("SIP Advantage", f"{bt.sip_vs_lumpsum_advantage_pct:+.1f}%",
                 "#1B9E4E" if bt.sip_vs_lumpsum_advantage_pct >= 0 else "#E2001A"),
                ("Worst Drawdown", f"{bt.worst_drawdown_pct:.1f}%", "#E2001A"),
            ]):
                col.markdown(f"""
                <div class="metric-card" style="border-top-color:{color};">
                    <div class="metric-value" style="color:{color};">{val}</div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

            if bt.annual_returns:
                ann_df = pd.DataFrame(bt.annual_returns)
                fig3 = go.Figure(go.Bar(
                    x=ann_df["date"], y=ann_df["annual_return_pct"],
                    marker_color=["#1B9E4E" if r >= 0 else "#E2001A"
                                  for r in ann_df["annual_return_pct"]],
                    hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
                ))
                fig3.add_hline(y=0, line_color="#888", line_width=1)
                fig3.update_layout(
                    title=dict(text="Year-by-Year Backtested Returns",
                               font=dict(family="'Playfair Display',serif", size=14, color="#1A1A1A")),
                    xaxis=dict(gridcolor="#F0F0F0", tickfont=CHART_FONT),
                    yaxis=dict(title="Return %", gridcolor="#F0F0F0",
                               tickfont=CHART_FONT, title_font=CHART_FONT),
                    font=CHART_FONT,
                    paper_bgcolor="white", plot_bgcolor="white",
                    margin=dict(t=40, b=40, l=55, r=20), height=230,
                )
                st.plotly_chart(fig3, use_container_width=True, key=f"chart_fire_backtest_{key_suffix}")

    with tab2:
        if roadmap.stress_results:
            st.markdown("**How your portfolio would have survived real market crashes** "
                        "*(SIP continued through crash)*")
            for key, stress in roadmap.stress_results.items():
                dd_color = "#E2001A" if stress.max_drawdown_pct < -30 else "#F39C12"
                rec_color = "#1B9E4E" if 0 < stress.recovery_months <= 12 else "#F39C12"
                with st.expander(f"📉 {stress.scenario_name}", expanded=True):
                    s1, s2, s3, s4 = st.columns(4)
                    rca_fmt = (f"₹{stress.sip_rupee_cost_advantage/1e5:.1f}L"
                               if stress.sip_rupee_cost_advantage >= 1e5
                               else f"₹{stress.sip_rupee_cost_advantage:,.0f}")
                    for col, (label, val, color) in zip([s1, s2, s3, s4], [
                        ("Max Drawdown", f"{stress.max_drawdown_pct:.1f}%", dd_color),
                        ("Recovery", f"{stress.recovery_months}mo" if stress.recovery_months else "N/A", rec_color),
                        ("RCA Advantage", rca_fmt, "#1B9E4E"),
                        ("Recovery Date", stress.recovery_date or "Ongoing", "#1A1A1A"),
                    ]):
                        col.markdown(f"""
                        <div class="metric-card" style="border-top-color:{color};">
                            <div class="metric-value" style="color:{color}; font-size:20px;">{val}</div>
                            <div class="metric-label">{label}</div>
                        </div>""", unsafe_allow_html=True)
                    st.info(
                        f"💡 By continuing SIPs through the {stress.scenario_name}, you accumulated "
                        f"extra units at depressed prices — a Rupee Cost Averaging advantage of "
                        f"**{rca_fmt}** vs stopping SIPs."
                    )
        else:
            st.info("Stress test data unavailable — network may be offline.")

    with tab3:
        if roadmap.monte_carlo_result:
            mc = roadmap.monte_carlo_result
            success_color = ("#1B9E4E" if mc.success_rate_pct >= 80
                             else "#F39C12" if mc.success_rate_pct >= 60 else "#E2001A")
            st.markdown(f"""
            <div style="text-align:center; padding:20px 0 10px;">
                <div style="font-size:11px; color:#888; text-transform:uppercase; letter-spacing:0.6px;">
                    Probability of Reaching FIRE Target
                </div>
                <div style="font-family:'Playfair Display',serif; font-size:56px;
                            font-weight:900; color:{success_color};">
                    {mc.success_rate_pct:.0f}%
                </div>
                <div style="font-size:13px; color:#555; margin-top:4px;">{mc.confidence_label}</div>
                <div style="font-size:11px; color:#999; margin-top:6px;">
                    {mc.num_simulations:,} simulations · log-normal returns · σ=16% (Nifty historical)
                </div>
            </div>""", unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            for col, (label, val, color) in zip([m1, m2, m3], [
                ("Pessimistic (P10)", _fmt(mc.p10_corpus), "#E2001A"),
                ("Median Outcome", _fmt(mc.median_corpus), "#1A1A1A"),
                ("Optimistic (P90)", _fmt(mc.p90_corpus), "#1B9E4E"),
            ]):
                col.markdown(f"""
                <div class="metric-card" style="border-top-color:{color};">
                    <div class="metric-value" style="color:{color};">{val}</div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

            MC_FONT = dict(color="#1A1A1A", family="'Source Sans 3', sans-serif", size=11)
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Scatter(
                x=["P10 (Bear)", "Median", "P90 (Bull)"],
                y=[mc.p10_corpus, mc.median_corpus, mc.p90_corpus],
                fill="tozeroy", fillcolor="rgba(226,0,26,0.08)",
                line=dict(color="#E2001A", width=2), mode="lines+markers",
                marker=dict(size=10, color=["#E2001A", "#1A1A1A", "#1B9E4E"]),
                hovertemplate="%{x}: ₹%{y:,.0f}<extra></extra>",
            ))
            fig_mc.add_hline(
                y=roadmap.required_corpus, line_dash="dot", line_color="#F0A500",
                annotation_text=f"Target: {_fmt(roadmap.required_corpus)}",
                annotation_position="bottom right",
                annotation_font=dict(color="#1A1A1A", size=11),
            )
            fig_mc.update_layout(
                title=dict(text="Corpus Distribution Across 1,000 Scenarios",
                           font=dict(family="'Playfair Display',serif", size=15, color="#1A1A1A")),
                xaxis=dict(gridcolor="#F0F0F0", tickfont=MC_FONT),
                yaxis=dict(title="Final Corpus (₹)", gridcolor="#F0F0F0",
                           tickformat=",.0f", tickfont=MC_FONT, title_font=MC_FONT),
                font=MC_FONT,
                paper_bgcolor="white", plot_bgcolor="white",
                margin=dict(t=45, b=40, l=70, r=20), height=290, showlegend=False,
            )
            st.plotly_chart(fig_mc, use_container_width=True, key=f"chart_fire_mc_{key_suffix}")
        else:
            st.info("Monte Carlo results unavailable.")


def render_market_data(result: qe.RollingReturnResult) -> None:
    """Render market data result as a clean metric card row."""
    cols = st.columns(4)
    metrics = [
        ("CAGR", f"{result.cagr_pct:.1f}%", "#1B9E4E" if result.cagr_pct > 0 else "#E2001A"),
        ("Total Return", f"{result.total_return_pct:.1f}%", "#1B9E4E" if result.total_return_pct > 0 else "#E2001A"),
        ("Max Drawdown", f"{result.max_drawdown_pct:.1f}%", "#E2001A"),
        ("Ann. Volatility", f"{result.volatility_annualised_pct:.1f}%", "#F39C12"),
    ]
    for col, (label, val, color) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-top-color:{color};">
                <div class="metric-value" style="color:{color};">{val}</div>
                <div class="metric-label">{label} ({result.period_label})</div>
            </div>
            """, unsafe_allow_html=True)
    st.caption(
        f"Source: yfinance · {result.ticker} · "
        f"{result.start_date} → {result.end_date} · "
        f"Start NAV: ₹{result.start_nav:,.2f} · End NAV: ₹{result.end_nav:,.2f}"
    )


def render_xray_report(report, key_suffix: str = "0") -> None:
    """Render the MF Portfolio X-Ray: XIRR table, overlap heatmap, expense drag chart."""
    XRAY_FONT = dict(color="#1A1A1A", family="'Source Sans 3', sans-serif", size=11)

    def _fmt(v):
        if abs(v) >= 1e7: return f"₹{v/1e7:.2f}Cr"
        return f"₹{abs(v)/1e5:.1f}L"

    tab1, tab2, tab3 = st.tabs(["📊 XIRR Report", "🔀 Fund Overlap", "💸 Expense Drag"])

    with tab1:
        on_track = report.total_current_value >= report.total_invested
        gain_color = "#1B9E4E" if on_track else "#E2001A"
        s1, s2, s3, s4 = st.columns(4)
        for col, (label, val, color) in zip([s1, s2, s3, s4], [
            ("Total Invested", _fmt(report.total_invested), "#1A1A1A"),
            ("Current Value", _fmt(report.total_current_value), gain_color),
            ("Portfolio XIRR", f"{report.portfolio_xirr_pct:.2f}%",
             "#1B9E4E" if report.portfolio_xirr_pct >= 10 else "#F39C12"),
            ("Abs. Return", f"{report.absolute_return_pct:.1f}%", gain_color),
        ]):
            col.markdown(f"""
            <div class="metric-card" style="border-top-color:{color};">
                <div class="metric-value" style="color:{color};">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Per-Fund XIRR Breakdown</div>',
                    unsafe_allow_html=True)
        fund_rows = []
        for f in sorted(report.fund_xirr_results, key=lambda x: -x.xirr_pct):
            fund_rows.append({
                "Fund": f.scheme_name[:48],
                "Category": f.category or "—",
                "Invested": _fmt(f.total_invested),
                "Current": _fmt(f.current_value),
                "Abs. Return": f"{f.absolute_return_pct:.1f}%",
                "XIRR": f"{f.xirr_pct:.2f}%",
                "TER": f"{f.expense_ratio_pct:.2f}%" if f.expense_ratio_pct else "—",
                "Period": f"{f.holding_period_years:.1f}Y",
            })
        st.dataframe(pd.DataFrame(fund_rows), hide_index=True, use_container_width=True,
                     height=min(40 + len(fund_rows) * 35, 400))

        if report.category_allocation:
            cat_df = pd.DataFrame([{"Category": k, "Value": v}
                                    for k, v in report.category_allocation.items()])
            fig_donut = go.Figure(go.Pie(
                labels=cat_df["Category"], values=cat_df["Value"], hole=0.55,
                marker_colors=["#E2001A","#1A1A1A","#F0A500","#2ECC71",
                                "#3498DB","#9B59B6","#1ABC9C","#E67E22"][:len(cat_df)],
                textinfo="label+percent",
                hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
            ))
            fig_donut.update_layout(
                title=dict(text="Category Allocation",
                           font=dict(family="'Playfair Display',serif", size=15, color="#1A1A1A")),
                paper_bgcolor="white", margin=dict(t=45, b=20, l=20, r=20), height=320,
                showlegend=False,
                font=XRAY_FONT,
            )
            st.plotly_chart(fig_donut, use_container_width=True, key=f"xray_donut_{key_suffix}")

        if report.high_ter_funds or report.concentrated_categories:
            st.markdown('<div class="section-heading">⚠️ Alerts</div>', unsafe_allow_html=True)
            for alert in report.high_ter_funds:
                st.warning(f"High TER: {alert}")
            for alert in report.concentrated_categories:
                st.warning(f"Concentration: {alert}")

        if report.recommendations:
            st.markdown('<div class="section-heading">💡 Recommendations</div>',
                        unsafe_allow_html=True)
            for rec in report.recommendations:
                st.info(rec)

    with tab2:
        if report.overlap_pairs:
            st.markdown("**Stock-level overlap between equity funds in your portfolio**")
            st.caption("Overlap >40% means you're paying for the same stocks twice.")
            overlap_rows = []
            for pair in report.overlap_pairs[:10]:
                overlap_rows.append({
                    "Fund A": pair.fund_a[:40], "Fund B": pair.fund_b[:40],
                    "Overlap": f"{pair.overlap_pct:.0f}%",
                    "Common Stocks": ", ".join(pair.common_stocks[:4]),
                    "Wtd Overlap": f"{pair.overlap_weight_pct:.1f}%",
                })
            st.dataframe(pd.DataFrame(overlap_rows), hide_index=True, use_container_width=True)

            top_pairs = report.overlap_pairs[:8]
            labels = [f"{p.fund_a[:20]}↔{p.fund_b[:20]}" for p in top_pairs]
            fig_ov = go.Figure(go.Bar(
                x=[p.overlap_pct for p in top_pairs], y=labels, orientation="h",
                marker_color=["#E2001A" if p.overlap_pct > 40
                              else "#F39C12" if p.overlap_pct > 25 else "#1B9E4E"
                              for p in top_pairs],
                hovertemplate="%{y}: %{x:.0f}%<extra></extra>",
            ))
            fig_ov.add_vline(x=40, line_dash="dot", line_color="#888",
                             annotation_text="40% threshold", annotation_position="top",
                             annotation_font=dict(color="#1A1A1A", size=10))
            fig_ov.update_layout(
                title=dict(text="Fund Overlap (% of top holdings shared)",
                           font=dict(family="'Playfair Display',serif", size=14, color="#1A1A1A")),
                xaxis=dict(title="Overlap %", range=[0, 100], gridcolor="#F0F0F0",
                           tickfont=XRAY_FONT, title_font=XRAY_FONT),
                yaxis=dict(autorange="reversed", tickfont=XRAY_FONT),
                font=XRAY_FONT, paper_bgcolor="white", plot_bgcolor="white",
                margin=dict(t=45, b=40, l=20, r=20), height=300,
            )
            st.plotly_chart(fig_ov, use_container_width=True, key=f"xray_overlap_{key_suffix}")
        else:
            st.info("No significant fund overlap detected.")

    with tab3:
        if report.expense_drag_results:
            st.markdown("**How much does your TER cost you over 20 years?**")
            st.caption("Assumes ₹5,000/month SIP per fund at 12% gross return.")
            total_drag = sum(d.drag_amount for d in report.expense_drag_results)
            st.markdown(f"""
            <div style="background:#FFF0F0; border-left:3px solid #E2001A;
                        padding:12px 16px; border-radius:6px; margin-bottom:16px;">
                <span style="font-family:'Playfair Display',serif; font-size:18px;
                             font-weight:700; color:#E2001A;">
                    Total 20-Year Expense Drag: {_fmt(total_drag)}
                </span>
                <div style="font-size:12px; color:#666; margin-top:4px;">
                    This is how much MORE you would have earned in Direct Plan / Zero-TER index funds.
                </div>
            </div>""", unsafe_allow_html=True)
            drag_rows = []
            for d in report.expense_drag_results:
                drag_rows.append({
                    "Fund": d.scheme_name[:45], "TER": f"{d.ter_pct:.2f}%",
                    "Est. Monthly SIP": f"₹{d.monthly_sip:,.0f}",
                    "Corpus WITH TER": _fmt(d.corpus_with_ter),
                    "Corpus WITHOUT TER": _fmt(d.corpus_without_ter),
                    "20Y Drag": _fmt(d.drag_amount), "Drag %": f"{d.drag_pct:.1f}%",
                })
            st.dataframe(pd.DataFrame(drag_rows), hide_index=True, use_container_width=True)
            fig_drag = go.Figure(go.Bar(
                x=[d.drag_amount for d in report.expense_drag_results],
                y=[d.scheme_name[:35] for d in report.expense_drag_results],
                orientation="h", marker_color="#E2001A",
                hovertemplate="%{y}: ₹%{x:,.0f}<extra></extra>",
            ))
            fig_drag.update_layout(
                title=dict(text="20-Year Expense Drag by Fund",
                           font=dict(family="'Playfair Display',serif", size=14, color="#1A1A1A")),
                xaxis=dict(title="Drag Amount (₹)", gridcolor="#F0F0F0", tickformat=",.0f",
                           tickfont=XRAY_FONT, title_font=XRAY_FONT),
                yaxis=dict(autorange="reversed", tickfont=XRAY_FONT),
                font=XRAY_FONT, paper_bgcolor="white", plot_bgcolor="white",
                margin=dict(t=45, b=40, l=20, r=20), height=300,
            )
            st.plotly_chart(fig_drag, use_container_width=True, key=f"xray_drag_{key_suffix}")
        else:
            st.info("Expense ratio data not available for your holdings.")


# ============================================================================ #
#  SECTION 6 — QUICK-START PROMPTS (shown when chat is empty)                  #
# ============================================================================ #

QUICK_PROMPTS = [
    "Check my money health score",
    "How much SIP do I need to retire at 50?",
    "What's the Nifty 50 5-year CAGR?",
    "I earn ₹80,000/month. Am I saving enough?",
    "Analyse my CAMS statement",
]

if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; padding: 40px 0 20px 0;">
        <div style="font-family:'Playfair Display',serif; font-size:26px;
                    font-weight:700; color:#1A1A1A;">
            Your AI Financial Co-pilot
        </div>
        <div style="color:#5A5A5A; margin-top:8px; font-size:15px;">
            Ask me anything about your money — I'll do the math.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick-start buttons
    cols = st.columns(len(QUICK_PROMPTS))
    for col, prompt in zip(cols, QUICK_PROMPTS):
        with col:
            if st.button(prompt, use_container_width=True, key=f"qs_{prompt[:20]}"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()

# ============================================================================ #
#  SECTION 7 — RENDER CHAT HISTORY                                             #
# ============================================================================ #

for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Each message stores its own result — no cross-message contamination
        msg_data = st.session_state.msg_results.get(idx, {})
        vtype = msg_data.get("type")
        result = msg_data.get("result")
        if result is None:
            continue
        if vtype == "health":
            render_health_score(result, key_suffix=f"h{idx}")
        elif vtype == "fire":
            render_fire_roadmap(result, key_suffix=f"h{idx}")
        elif vtype == "sip":
            render_sip_chart(result, key_suffix=f"h{idx}")
        elif vtype == "market":
            render_market_data(result)
        elif vtype == "xray":
            render_xray_report(result, key_suffix=f"h{idx}")

# ============================================================================ #
#  SECTION 8 — MAIN ROUTING ENGINE (the core of app.py)                       #
# ============================================================================ #

def _format_inr(amount: float) -> str:
    """Format INR amounts with lakh/crore notation."""
    if abs(amount) >= 1e7:
        return f"₹{amount/1e7:.2f} Cr"
    elif abs(amount) >= 1e5:
        return f"₹{amount/1e5:.1f}L"
    else:
        return f"₹{amount:,.0f}"


def _route_and_respond(user_input: str) -> None:
    """
    Core routing function. Called once per user message.
    Detects intent → extracts params → calls quant engine → generates response.
    All mutations go to st.session_state; return value is None.
    """
    history_str = format_history(st.session_state.messages[:-1])  # Exclude current message

    # ── Step 1: Detect Intent ───────────────────────────────────────────────── #
    try:
        with st.spinner("Understanding your question..."):
            if forced_intent:
                # User explicitly chose a module via sidebar
                from llm_orchestrator import IntentResult
                intent_result = IntentResult(
                    intent=forced_intent,
                    confidence=1.0,
                    missing_info=[],
                    reasoning="User selected module explicitly via sidebar.",
                )
            else:
                intent_result = detect_intent(user_input, history_str)
    except RuntimeError as e:
        st.error(f"Could not process your request: {e}")
        return

    intent = intent_result.intent

    # ── Deterministic override: strong HEALTH_SCORE signals ─────────────────── #
    # If the message contains income + any of (expenses/emergency/insurance/debt),
    # always route to HEALTH_SCORE regardless of LLM classification.
    # This prevents CAMS context bias from hijacking clear health score queries.
    msg_lower = user_input.lower()
    has_income = any(k in msg_lower for k in ["earn", "income", "salary", "per month"])
    has_health_signal = any(k in msg_lower for k in [
        "expense", "emergency", "insurance", "debt", "emi", "saving"
    ])
    if has_income and has_health_signal and intent != Intent.HEALTH_SCORE:
        logger.info("Deterministic override: %s → HEALTH_SCORE (income+health signals)", intent)
        intent = Intent.HEALTH_SCORE

    logger.info("Routing to intent: %s", intent)

    # ── Step 2: Handle CLARIFY (not enough info yet) ─────────────────────────── #
    if intent == Intent.CLARIFY or (
        intent != Intent.GENERAL_QUERY and len(intent_result.missing_info) > 2
    ):
        clarification = generate_clarification_request(
            user_input, intent_result.missing_info, intent
        )
        _add_assistant_message(clarification)
        return

    # ── Step 3: GENERAL_QUERY (no math needed) ──────────────────────────────── #
    if intent == Intent.GENERAL_QUERY:
        with st.spinner("Thinking..."):
            response = generate_general_response(user_input, history_str)
        _add_assistant_message(response)
        return

    # ── Step 4: Intent-specific routing ────────────────────────────────────── #

    # ── 4a. HEALTH_SCORE ────────────────────────────────────────────────────── #
    if intent == Intent.HEALTH_SCORE:
        with st.spinner("Calculating your Money Health Score..."):
            try:
                params = extract_health_params(user_input, history_str)
                health_result = qe.calculate_money_health_score(
                    monthly_income=params.monthly_income,
                    monthly_expenses=params.monthly_expenses,
                    emergency_fund=params.emergency_fund,
                    total_insurance_cover=params.total_insurance_cover,
                    total_debt_emi=params.total_debt_emi,
                    equity_pct=params.equity_pct,
                    debt_pct=params.debt_pct,
                    gold_pct=params.gold_pct,
                    other_pct=params.other_pct,
                    epf_ppf_nps_monthly=params.epf_ppf_nps_monthly,
                    tax_saving_investments=params.tax_saving_investments,
                    gross_annual_income=params.gross_annual_income,
                )
                st.session_state.health_result = health_result
                narrative = generate_response(
                    user_input,
                    json.dumps(health_result, indent=2),
                    context=f"Monthly income: {_format_inr(params.monthly_income)}",
                )
                _add_assistant_message(narrative, viz_type="health", result=health_result)
            except (ValueError, RuntimeError) as e:
                _add_assistant_message(
                    f"I ran into an issue calculating your score: **{e}**\n\n"
                    "Could you double-check the numbers you provided?"
                )

    # ── 4b. SIP_PROJECTION ──────────────────────────────────────────────────── #
    elif intent == Intent.SIP_PROJECTION:
        with st.spinner("Projecting your SIP growth..."):
            try:
                params = extract_sip_params(user_input, history_str)
                sip_result = qe.project_sip_corpus(
                    monthly_sip=params.monthly_sip,
                    years=params.years,
                    assumed_cagr_pct=params.assumed_cagr_pct,
                    target_corpus=params.target_corpus,
                    step_up_pct=params.step_up_pct,
                )
                st.session_state.sip_result = sip_result
                narrative = generate_response(
                    user_input,
                    json.dumps({
                        "monthly_sip": sip_result.monthly_sip,
                        "years": sip_result.years,
                        "assumed_cagr_pct": sip_result.assumed_cagr_pct,
                        "projected_corpus": sip_result.projected_corpus,
                        "target_corpus": sip_result.target_corpus,
                        "shortfall_surplus": sip_result.shortfall_surplus,
                        "step_up_pct": params.step_up_pct,
                    }, indent=2),
                )
                _add_assistant_message(narrative, viz_type="sip", result=sip_result)
            except (ValueError, RuntimeError) as e:
                _add_assistant_message(
                    f"I couldn't calculate the SIP projection: **{e}**\n\n"
                    "Please check your inputs and try again."
                )

    # ── 4c. MARKET_DATA ─────────────────────────────────────────────────────── #
    elif intent == Intent.MARKET_DATA:
        with st.spinner("Fetching live market data..."):
            try:
                params = extract_market_params(user_input, history_str)
                market_result = qe.fetch_historical_rolling_return(
                    params.ticker_or_alias, params.period
                )
                st.session_state.market_result = market_result
                narrative = generate_response(
                    user_input,
                    json.dumps(asdict(market_result), indent=2),
                )
                _add_assistant_message(narrative, viz_type="market", result=market_result)
            except (ValueError, RuntimeError) as e:
                _add_assistant_message(
                    f"Couldn't fetch market data: **{e}**\n\n"
                    "Check the ticker or try 'Nifty 50' / 'Sensex'."
                )

    # ── 4d. FIRE_PLANNER ────────────────────────────────────────────────────── #
    elif intent == Intent.FIRE_PLANNER:
        with st.spinner("Building your backtest-validated FIRE roadmap... (fetching live Nifty data)"):
            try:
                from fire_planner import build_fire_roadmap, format_roadmap_for_llm
                params = extract_fire_params(user_input, history_str)

                roadmap = build_fire_roadmap(
                    current_age=params.current_age,
                    retirement_age=params.retirement_age,
                    monthly_income=params.monthly_income,
                    monthly_expenses=params.monthly_expenses,
                    current_savings=params.current_savings,
                    user_monthly_sip=params.monthly_sip,  # user's ACTUAL SIP
                    target_corpus=params.target_corpus,
                    assumed_inflation_pct=params.assumed_inflation_pct,
                    assumed_return_pct=params.assumed_return_pct,
                    step_up_pct=params.step_up_pct,
                    run_backtest=True,
                    run_stress_test=True,
                    run_monte_carlo=True,
                )
                st.session_state.fire_result = roadmap
                # NOTE: We do NOT overwrite sip_result here — that would corrupt
                # any standalone SIP projection already in the chat history.

                narrative = generate_response(
                    user_input,
                    format_roadmap_for_llm(roadmap),
                    context=(
                        f"Monthly income: {_format_inr(params.monthly_income)}. "
                        f"Current savings: {_format_inr(params.current_savings)}."
                    ),
                )
                _add_assistant_message(narrative, viz_type="fire", result=roadmap)
            except (ValueError, RuntimeError) as e:
                _add_assistant_message(
                    f"I hit an issue building your FIRE roadmap: **{e}**\n\n"
                    "Please share your age, monthly income, and target retirement age."
                )

    # ── 4e. MF_XRAY ─────────────────────────────────────────────────────────── #
    elif intent == Intent.MF_XRAY:
        if st.session_state.cams_data is None:
            _add_assistant_message(
                "To run the MF Portfolio X-Ray, please upload your **CAMS Statement PDF** "
                "using the sidebar uploader.\n\n"
                "🔒 Your file is processed **100% locally** — no data leaves your device."
            )
        else:
            with st.spinner("Running X-Ray analysis (XIRR · Overlap · Expense Drag)..."):
                try:
                    from mf_xray import build_xray_report, format_xray_for_llm
                    report = build_xray_report(st.session_state.cams_data)
                    st.session_state.xray_report = report
                    narrative = generate_response(
                        user_input,
                        format_xray_for_llm(report),
                        context=(
                            f"Investor: {report.investor_name}. "
                            f"Portfolio XIRR: {report.portfolio_xirr_pct:.2f}%. "
                            f"Completed in {report.analysis_duration_sec:.2f}s."
                        ),
                    )
                    _add_assistant_message(narrative, viz_type="xray", result=report)
                except (ValueError, RuntimeError) as e:
                    _add_assistant_message(
                        f"X-Ray analysis failed: **{e}**\n\n"
                        "Please ensure the uploaded file is a valid CAMS/KFintech statement."
                    )


def _add_assistant_message(
    content: str,
    viz_type: Optional[str] = None,
    result=None,
) -> None:
    """
    Append an assistant message and render it immediately.
    Stores the viz result keyed by message index so history replay
    always shows each message's own result, never a later one.
    """
    msg = {"role": "assistant", "content": content}
    st.session_state.messages.append(msg)

    # Store result against this message's index
    if viz_type and result is not None:
        msg_idx = len(st.session_state.messages) - 1
        st.session_state.msg_results[msg_idx] = {"type": viz_type, "result": result}

    with st.chat_message("assistant"):
        st.markdown(content)
        suffix = f"new{len(st.session_state.messages)}"
        if viz_type == "health" and result is not None:
            render_health_score(result, key_suffix=suffix)
        elif viz_type == "fire" and result is not None:
            render_fire_roadmap(result, key_suffix=suffix)
        elif viz_type == "sip" and result is not None:
            render_sip_chart(result, key_suffix=suffix)
        elif viz_type == "market" and result is not None:
            render_market_data(result)
        elif viz_type == "xray" and result is not None:
            render_xray_report(result, key_suffix=suffix)


# ============================================================================ #
#  SECTION 9 — CHAT INPUT (must be last in file — Streamlit requirement)       #
# ============================================================================ #

if user_input := st.chat_input("Ask about your finances... (e.g. 'I earn ₹75k/month, am I on track?')"):
    # Display user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Route and respond
    _route_and_respond(user_input)
