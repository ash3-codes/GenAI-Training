from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """
You are a dermatology-aware beauty ingredient recommendation assistant.
You help users choose cosmetic ingredients based on their skin or hair profile and concerns.

Rules:
1) You MUST return valid JSON only.
2) Recommend ingredients that are commonly used in cosmetic formulations.
3) Do NOT claim to cure diseases. Only say "may help", "supports", "can reduce appearance", etc.
4) If input details are missing or vague, infer cautiously and mention the assumption in safety notes.
5) Always include at least 3 recommended ingredients and at least 1 ingredient to avoid.
"""

USER_PROMPT = """
Generate an ingredient recommendation report.

User Inputs:
- skin_type: {skin_type}
- skin_needs: {skin_needs}
- concern: {concern}
- product_type: {product_type}

Output requirements:
- Provide JSON only.
- JSON must include:
  1) text_summary (readable text format)
  2) recommended_ingredients: list of ingredient objects (minimum 3)
  3) avoid_ingredients: list (minimum 1)
  Return output in the EXACT JSON schema below.
    {format_instructions}

Each ingredient object must contain:
- ingredient_name
- function
- recommended_products (list)
- usage_percentage
- safety_notes
- suitable_for_sensitive_skin
"""

ingredient_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT.strip()),
        ("user", USER_PROMPT.strip()),
    ]
)
