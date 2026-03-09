# llm/prompt_templates.py

from config.constants import SYSTEM_GUARDRAIL_MESSAGE


def build_prompt(context: str, user_question: str) -> str:

    prompt = f"""
{SYSTEM_GUARDRAIL_MESSAGE}

========================
POLICY CONTEXT:
========================

{context}

========================
USER QUESTION:
========================

{user_question}

========================
INSTRUCTIONS:
- Answer using only the policy context.
- Provide concise and clear response.
- Cite each relevant statement.
- Do not fabricate policy information.
"""

    return prompt