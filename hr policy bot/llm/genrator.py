# llm/generator.py

from llm.azure_config import get_llm
from llm.prompt_templates import build_prompt


class HRPolicyGenerator:

    def __init__(self):
        self.llm = get_llm()

    def generate(self, context: str, question: str) -> str:

        prompt = build_prompt(context, question)

        response = self.llm.invoke(prompt)

        return response.content