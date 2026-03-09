# app.py

import streamlit as st
from rag.orchestrator import HRPolicyOrchestrator

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📘"
)

st.title("📘 Internal HR Policy FAQ Bot")

# ---------------------------------------------------------
# Initialize Orchestrator + Chat History
# ---------------------------------------------------------

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = HRPolicyOrchestrator()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------------------------------------
# Display Chat History
# ---------------------------------------------------------

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------------------------------
# User Input
# ---------------------------------------------------------

user_input = st.chat_input("Ask your HR policy question...")

if user_input:

    # Store user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            response = st.session_state.orchestrator.ask(user_input)

            answer = response["answer"]
            sources = response.get("sources", [])

            st.markdown(answer)

            if sources:
                st.markdown("---")
                st.markdown("**Sources:**")

                unique_sources = {
                    (s.get("doc_name"), s.get("page_number"))
                    for s in sources
                }

                for doc_name, page_number in unique_sources:
                    st.markdown(
                        f"- {doc_name} (Page {page_number})"
                    )

    # Store assistant message
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer
    })