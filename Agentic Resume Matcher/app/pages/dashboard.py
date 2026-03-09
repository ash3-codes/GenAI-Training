# -*- coding: utf-8 -*-
"""
app/pages/dashboard.py
----------------------
HR Dashboard page -- previous searches, JD upload history, resume rankings.
Shown in Streamlit's multi-page sidebar as "Dashboard".

Tracks every query run in this session and lets the HR user:
  - Review previous JD searches and their top results
  - Re-open any past result set
  - See a summary table of all candidates ranked across searches
  - Export rankings as CSV
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")

import streamlit as st

st.set_page_config(
    page_title="Dashboard | Resume Matcher",
    page_icon=":bar_chart:",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    "<style>"
    "@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500"
    "&family=Syne:wght@700;800&display=swap');"
    ":root{--bg:#0d0f14;--surface:#13151c;--border:#1e2130;"
    "--accent:#5b6af5;--accent2:#38d9a9;--danger:#f06595;"
    "--text:#e8eaf0;--muted:#6b7280;}"
    "html,body,[class*='css']{background:var(--bg)!important;"
    "color:var(--text)!important;font-family:'Syne',sans-serif!important;}"
    "[data-testid='stSidebar']{background:var(--surface)!important;"
    "border-right:1px solid var(--border)!important;}"
    ".top-bar{display:flex;align-items:center;gap:12px;padding:0 0 20px;"
    "border-bottom:1px solid var(--border);margin-bottom:24px;}"
    ".top-bar h1{font-family:'Syne',sans-serif;font-size:1.35rem;font-weight:800;"
    "letter-spacing:-0.02em;color:var(--text);margin:0;}"
    ".badge{font-family:'DM Mono',monospace;font-size:0.68rem;color:var(--accent2);"
    "background:rgba(56,217,169,0.08);border:1px solid rgba(56,217,169,0.18);"
    "border-radius:4px;padding:2px 8px;margin-left:auto;}"
    ".shdr{font-family:'DM Mono',monospace;font-size:0.65rem;color:var(--muted);"
    "text-transform:uppercase;letter-spacing:0.12em;"
    "border-bottom:1px solid var(--border);padding-bottom:7px;margin:18px 0 12px;}"
    ".hcard{background:var(--surface);border:1px solid var(--border);"
    "border-radius:10px;padding:14px 16px;margin-bottom:8px;"
    "cursor:pointer;transition:border-color 0.2s;}"
    ".hcard:hover{border-color:var(--accent);}"
    ".hcard-title{font-family:'Syne',sans-serif;font-size:0.9rem;font-weight:700;"
    "color:var(--text);margin-bottom:3px;}"
    ".hcard-meta{font-family:'DM Mono',monospace;font-size:0.62rem;color:var(--muted);}"
    ".scard{background:var(--surface);border:1px solid var(--border);"
    "border-radius:10px;padding:18px;text-align:center;}"
    ".sval{font-family:'DM Mono',monospace;font-size:1.8rem;font-weight:500;"
    "color:var(--accent2);line-height:1;}"
    ".slbl{font-family:'DM Mono',monospace;font-size:0.6rem;color:var(--muted);"
    "text-transform:uppercase;letter-spacing:0.1em;margin-top:5px;}"
    ".rtrow{font-family:'DM Mono',monospace;font-size:0.68rem;"
    "padding:5px 0;border-bottom:1px solid var(--border);"
    "display:grid;grid-template-columns:2fr 1.5fr 0.7fr 0.7fr 0.7fr 0.7fr;"
    "gap:8px;align-items:center;}"
    ".rthead{color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;"
    "font-size:0.6rem;}"
    ".chip-jd{display:inline-block;font-family:'DM Mono',monospace;"
    "font-size:0.6rem;background:rgba(91,106,245,0.1);color:var(--accent);"
    "border:1px solid rgba(91,106,245,0.2);border-radius:4px;"
    "padding:1px 7px;margin:1px 2px;}"
    ".stButton>button{font-family:'DM Mono',monospace!important;"
    "background:var(--accent)!important;color:white!important;"
    "border:none!important;border-radius:6px!important;font-size:0.72rem!important;}"
    "#MainMenu,footer,header{visibility:hidden;}"
    "::-webkit-scrollbar{width:4px;}"
    "::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px;}"
    "</style>",
    unsafe_allow_html=True,
)

# ── Auth guard ────────────────────────────────────────────────────────────────
if not st.session_state.get("authenticated"):
    st.markdown(
        "<div style='text-align:center;padding:60px 0;'>"
        "<div style='font-family:DM Mono,monospace;font-size:0.8rem;"
        "color:#6b7280;'>Please sign in via the Account page first.</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# ── Session state init ────────────────────────────────────────────────────────
if "search_history" not in st.session_state:
    st.session_state["search_history"] = []

# Append latest query result to history whenever it changes
if "qr" in st.session_state:
    qr      = st.session_state["qr"]
    parsed  = qr.get("parsed_jd") or {}
    summary = qr.get("results_summary") or {}
    title   = str(parsed.get("title") or "Untitled JD")
    ts      = st.session_state.get("elapsed_ts", "")

    # Only append if it's a new result (different JD title or empty history)
    history = st.session_state["search_history"]
    if not history or history[-1].get("title") != title:
        import datetime
        history.append({
            "title":        title,
            "timestamp":    datetime.datetime.now().strftime("%H:%M:%S"),
            "elapsed":      st.session_state.get("elapsed", 0),
            "n_candidates": summary.get("total_candidates", 0),
            "top_candidate":summary.get("top_candidate"),
            "top_score":    summary.get("top_score", 0),
            "avg_score":    summary.get("avg_score", 0),
            "skill_cov":    summary.get("skill_coverage", 0),
            "required_skills": [s.get("skill") or "" for s in (parsed.get("required_skills") or [])],
            "final_scores": qr.get("final_scores") or [],
        })

history = st.session_state["search_history"]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='top-bar'>"
    "<h1>Dashboard</h1>"
    "<span class='badge'>SEARCH HISTORY</span>"
    "</div>",
    unsafe_allow_html=True,
)

# ── Summary stats row ─────────────────────────────────────────────────────────
total_searches   = len(history)
total_candidates = sum(h.get("n_candidates", 0) for h in history)
avg_top_score    = (
    round(sum(h.get("top_score", 0) for h in history) / total_searches, 2)
    if total_searches else 0
)

c1, c2, c3, c4 = st.columns(4)
for col, val, lbl in [
    (c1, total_searches,   "Searches This Session"),
    (c2, total_candidates, "Total Candidates Seen"),
    (c3, str(round(avg_top_score * 100)) + "%", "Avg Top Score"),
    (c4, str(1 if "ir" in st.session_state else 0), "Ingestions Run"),
]:
    col.markdown(
        "<div class='scard'>"
        "<div class='sval'>" + str(val) + "</div>"
        "<div class='slbl'>" + lbl + "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

if not history:
    st.markdown(
        "<div style='text-align:center;padding:60px 20px;color:#6b7280;'>"
        "<div style='font-family:DM Mono,monospace;font-size:0.75rem;"
        "letter-spacing:0.08em;'>NO SEARCHES YET</div>"
        "<div style='font-family:DM Mono,monospace;font-size:0.63rem;"
        "margin-top:8px;'>Run a JD match from the main page to see results here.</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# ── Previous Searches ─────────────────────────────────────────────────────────
left, right = st.columns([1, 1.6], gap="large")

with left:
    st.markdown("<div class='shdr'>Previous Searches</div>", unsafe_allow_html=True)

    selected_idx = st.session_state.get("dash_selected", len(history) - 1)

    for i, h in enumerate(reversed(history)):
        real_idx  = len(history) - 1 - i
        is_active = real_idx == selected_idx
        border    = "var(--accent)" if is_active else "var(--border)"
        skills    = h.get("required_skills") or []
        skill_str = "".join("<span class='chip-jd'>" + s + "</span>" for s in skills[:4])
        n_sk_more = len(skills) - 4
        if n_sk_more > 0:
            skill_str += "<span class='chip-jd'>+" + str(n_sk_more) + "</span>"

        top_name = str(h.get("top_candidate") or "None")
        top_sc   = str(round((h.get("top_score") or 0) * 100)) + "%"
        n_cands  = str(h.get("n_candidates") or 0)
        ts       = str(h.get("timestamp") or "")
        t_elapsed = str(round(h.get("elapsed") or 0, 1)) + "s"

        st.markdown(
            "<div class='hcard' style='border-color:" + border + ";'>"
            "<div class='hcard-title'>" + str(h.get("title", "")) + "</div>"
            "<div class='hcard-meta'>"
            + ts + " &middot; " + n_cands + " candidates &middot; " + t_elapsed
            + "</div>"
            "<div style='margin:5px 0;'>" + skill_str + "</div>"
            "<div class='hcard-meta'>Top: " + top_name + " (" + top_sc + ")</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        if st.button("View", key="view_" + str(real_idx), use_container_width=True):
            st.session_state["dash_selected"] = real_idx
            st.rerun()

with right:
    st.markdown("<div class='shdr'>Candidate Rankings</div>", unsafe_allow_html=True)

    selected_idx = st.session_state.get("dash_selected", len(history) - 1)
    if selected_idx >= len(history):
        selected_idx = len(history) - 1

    h        = history[selected_idx]
    final    = h.get("final_scores") or []
    req_set  = {s.lower() for s in (h.get("required_skills") or [])}

    if not final:
        st.info("No candidates in this search result.")
    else:
        # Table header
        st.markdown(
            "<div class='rtrow rthead'>"
            "<span>Candidate</span>"
            "<span>File</span>"
            "<span>Final</span>"
            "<span>ATS</span>"
            "<span>Semantic</span>"
            "<span>Exp</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        for c in final:
            name    = str(c.get("name") or "Unknown")
            payload = c.get("payload") or {}
            fname   = str(payload.get("file_name") or "")
            fscore  = float(c.get("final_score") or 0)
            ats     = float(c.get("ats_score") or 0)
            sem     = float(c.get("semantic_score") or 0)
            exp_mo  = float(payload.get("total_experience_months") or 0)
            exp_yr  = str(round(exp_mo / 12, 1)) + "y"
            rank    = int(c.get("final_rank") or 0)

            sc_col  = "#38d9a9" if fscore > 0.7 else ("#5b6af5" if fscore > 0.5 else "#6b7280")
            rank_str = "#" + str(rank)

            st.markdown(
                "<div class='rtrow'>"
                "<span><strong>" + name + "</strong>"
                " <span style='color:#6b7280;font-size:0.6rem;'>" + rank_str + "</span></span>"
                "<span style='color:#6b7280;font-size:0.62rem;'>" + fname + "</span>"
                "<span style='color:" + sc_col + ";font-weight:500;'>"
                + str(round(fscore * 100)) + "%</span>"
                "<span>" + str(round(ats * 100)) + "%</span>"
                "<span>" + str(round(sem * 100)) + "%</span>"
                "<span>" + exp_yr + "</span>"
                "</div>",
                unsafe_allow_html=True,
            )

        # CSV export
        st.markdown("<br>", unsafe_allow_html=True)
        import io, csv

        def build_csv(candidates):
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow([
                "rank", "name", "file_name", "final_score", "ats_score",
                "semantic_score", "rerank_score", "experience_years", "skills",
            ])
            for c in candidates:
                p    = c.get("payload") or {}
                exp  = round(float(p.get("total_experience_months") or 0) / 12, 1)
                sks  = "|".join(p.get("skills") or [])
                writer.writerow([
                    c.get("final_rank"),
                    c.get("name"),
                    p.get("file_name"),
                    round(float(c.get("final_score") or 0), 3),
                    round(float(c.get("ats_score") or 0), 3),
                    round(float(c.get("semantic_score") or 0), 3),
                    round(float(c.get("rerank_score") or 0), 3),
                    exp,
                    sks,
                ])
            return buf.getvalue()

        jd_slug = h.get("title", "results").lower().replace(" ", "_")[:30]
        csv_str = build_csv(final)
        st.download_button(
            label="Export CSV",
            data=csv_str,
            file_name=jd_slug + "_rankings.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ── JD Upload History ─────────────────────────────────────────────────────────
st.markdown("<div class='shdr'>JD Upload History (This Session)</div>",
            unsafe_allow_html=True)

if not history:
    st.markdown(
        "<div style='font-family:DM Mono,monospace;font-size:0.68rem;"
        "color:#6b7280;'>No JDs searched yet.</div>",
        unsafe_allow_html=True,
    )
else:
    rows_html = ""
    for i, h in enumerate(history):
        skills = h.get("required_skills") or []
        chips  = "".join("<span class='chip-jd'>" + s + "</span>" for s in skills[:6])
        n_cands = str(h.get("n_candidates") or 0)
        rows_html += (
            "<div style='display:grid;grid-template-columns:2fr 3fr 0.8fr;"
            "gap:8px;padding:5px 0;border-bottom:1px solid #1e2130;"
            "font-family:DM Mono,monospace;font-size:0.68rem;"
            "align-items:center;'>"
            "<span style='font-weight:600;color:#e8eaf0;'>"
            + str(h.get("title", "")) + "</span>"
            "<span>" + chips + "</span>"
            "<span style='color:#6b7280;'>" + n_cands + " results</span>"
            "</div>"
        )

    st.markdown(
        "<div style='background:var(--surface);border:1px solid var(--border);"
        "border-radius:10px;padding:12px 16px;'>"
        "<div style='display:grid;grid-template-columns:2fr 3fr 0.8fr;"
        "gap:8px;padding-bottom:5px;border-bottom:1px solid #1e2130;"
        "font-family:DM Mono,monospace;font-size:0.6rem;"
        "color:#6b7280;text-transform:uppercase;letter-spacing:0.08em;'>"
        "<span>JD Title</span><span>Required Skills</span><span>Results</span>"
        "</div>"
        + rows_html
        + "</div>",
        unsafe_allow_html=True,
    )