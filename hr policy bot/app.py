# app.py

import streamlit as st
from rag.orchestrator import HRPolicyOrchestrator

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📘",
    layout="wide"
)

st.title("📘 Internal HR Policy Assistant")

# ---------------------------------------------------------
# Initialize RAG + Session Memory
# ---------------------------------------------------------

if "rag" not in st.session_state:
    st.session_state.rag = HRPolicyOrchestrator()

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

user_input = st.chat_input("Ask a question about HR policies...")

if user_input:

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })

    # Get response from RAG
    with st.chat_message("assistant"):
        with st.spinner("Searching policy documents..."):
            response = st.session_state.rag.ask(user_input)

            answer = response["answer"]
            sources = response["sources"]

            st.markdown(answer)

            if sources:
                st.markdown("---")
                st.markdown("**Sources:**")

                unique_sources = {
                    (s["doc_name"], s["page_number"])
                    for s in sources
                }

                for doc_name, page_number in unique_sources:
                    st.markdown(
                        f"- {doc_name} (Page {page_number})"
                    )

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer
    })