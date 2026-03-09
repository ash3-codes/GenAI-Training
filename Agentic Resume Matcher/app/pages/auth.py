# -*- coding: utf-8 -*-
"""
app/pages/app.py
----------------
HR Login / Account settings page.
Shown in Streamlit's multi-page sidebar as "Account".

Handles:
  - Login form (if not authenticated)
  - Account info (if authenticated)
  - Password change (local session only)
  - Sign out
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")

import streamlit as st

st.set_page_config(
    page_title="Account | Resume Matcher",
    page_icon=":bust_in_silhouette:",
    layout="centered",
)

# ── shared CSS ────────────────────────────────────────────────────────────────
st.markdown(
    "<style>"
    "@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500"
    "&family=Syne:wght@700;800&display=swap');"
    ":root{--bg:#0d0f14;--surface:#13151c;--border:#1e2130;"
    "--accent:#5b6af5;--accent2:#38d9a9;--text:#e8eaf0;--muted:#6b7280;}"
    "html,body,[class*='css']{background:var(--bg)!important;"
    "color:var(--text)!important;font-family:'Syne',sans-serif!important;}"
    "[data-testid='stSidebar']{background:var(--surface)!important;"
    "border-right:1px solid var(--border)!important;}"
    ".stButton>button{font-family:'DM Mono',monospace!important;"
    "background:var(--accent)!important;color:white!important;"
    "border:none!important;border-radius:6px!important;}"
    ".stTextInput input{background:var(--surface)!important;"
    "border:1px solid var(--border)!important;color:var(--text)!important;"
    "font-family:'DM Mono',monospace!important;border-radius:6px!important;}"
    "#MainMenu,footer,header{visibility:hidden;}"
    ".panel{background:var(--surface);border:1px solid var(--border);"
    "border-radius:10px;padding:24px 28px;margin-bottom:16px;}"
    ".lbl{font-family:'DM Mono',monospace;font-size:0.65rem;color:var(--muted);"
    "text-transform:uppercase;letter-spacing:0.1em;margin-bottom:3px;}"
    ".val{font-size:0.9rem;font-weight:600;margin-bottom:12px;}"
    "</style>",
    unsafe_allow_html=True,
)

USERS = {"admin": "hr2025", "recruiter": "match123"}

# ── Not logged in: show login form ────────────────────────────────────────────
if not st.session_state.get("authenticated"):
    st.markdown(
        "<div style='text-align:center;padding:40px 0 28px;'>"
        "<h2 style='font-family:Syne,sans-serif;font-weight:800;"
        "letter-spacing:-0.02em;'>Sign In</h2>"
        "<p style='font-family:DM Mono,monospace;font-size:0.68rem;"
        "color:#6b7280;'>Resume Matcher &mdash; HR Platform</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    username = st.text_input("Username", placeholder="username")
    password = st.text_input("Password", type="password", placeholder="password")

    if st.button("Sign In", use_container_width=True):
        if USERS.get(username) == password:
            st.session_state["authenticated"] = True
            st.session_state["username"]      = username
            st.success("Signed in as " + username + ". Navigate using the sidebar.")
            st.rerun()
        else:
            st.error("Invalid username or password.")

    st.markdown(
        "<div style='font-family:DM Mono,monospace;font-size:0.63rem;"
        "color:#6b7280;text-align:center;margin-top:16px;'>"
        "Default accounts: admin / hr2025 &nbsp;&middot;&nbsp; recruiter / match123"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Logged in: show account info ──────────────────────────────────────────────
else:
    username = st.session_state.get("username", "unknown")

    st.markdown(
        "<h2 style='font-family:Syne,sans-serif;font-weight:800;"
        "letter-spacing:-0.02em;margin-bottom:20px;'>Account</h2>",
        unsafe_allow_html=True,
    )

    # Profile panel
    st.markdown(
        "<div class='panel'>"
        "<div class='lbl'>Signed in as</div>"
        "<div class='val'>" + username + "</div>"
        "<div class='lbl'>Role</div>"
        "<div class='val'>" + ("Administrator" if username == "admin" else "Recruiter") + "</div>"
        "<div class='lbl'>Session stats</div>"
        "<div style='font-family:DM Mono,monospace;font-size:0.75rem;color:#6b7280;'>"
        "Searches this session: "
        + str(len(st.session_state.get("search_history", [])))
        + "&nbsp;&middot;&nbsp;Ingestions this session: "
        + str(1 if "ir" in st.session_state else 0)
        + "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Qdrant connection status
    st.markdown(
        "<div style='font-family:DM Mono,monospace;font-size:0.65rem;"
        "color:#6b7280;text-transform:uppercase;letter-spacing:0.1em;"
        "border-bottom:1px solid #1e2130;padding-bottom:7px;margin:18px 0 12px;'>"
        "System Status</div>",
        unsafe_allow_html=True,
    )

    try:
        from config.settings import (
            get_qdrant_client, QDRANT_RESUME_COLLECTION,
            AZURE_CHAT_DEPLOYMENT, AZURE_EMBEDDING_DEPLOYMENT,
        )
        client = get_qdrant_client()
        r_cnt  = client.get_collection(QDRANT_RESUME_COLLECTION).points_count
        rows = [
            ("Qdrant",      "Connected", "#38d9a9"),
            ("Resumes indexed", str(r_cnt), "#38d9a9"),
            ("LLM",         AZURE_CHAT_DEPLOYMENT, "#38d9a9"),
            ("Embeddings",  AZURE_EMBEDDING_DEPLOYMENT, "#38d9a9"),
        ]
    except Exception as e:
        rows = [("Qdrant", "ERROR: " + str(e)[:60], "#f06595")]

    for lbl, val, col in rows:
        st.markdown(
            "<div style='display:flex;justify-content:space-between;"
            "font-family:DM Mono,monospace;font-size:0.72rem;"
            "padding:5px 0;border-bottom:1px solid #1e2130;'>"
            "<span style='color:#6b7280;'>" + lbl + "</span>"
            "<span style='color:" + col + ";'>" + val + "</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Session Data", use_container_width=True):
            for key in ["qr", "ir", "elapsed", "top_k", "search_history"]:
                st.session_state.pop(key, None)
            st.success("Session data cleared.")
    with col2:
        if st.button("Sign Out", use_container_width=True):
            st.session_state["authenticated"] = False
            st.session_state.pop("username", None)
            st.rerun()