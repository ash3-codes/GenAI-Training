# app/post_chain.py
import json
from typing import Dict, Any

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

from prompts import PLANNER_PROMPT, WRITER_PROMPT


def build_post_chain():
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.6
    )

    parser = StrOutputParser()

    planner_chain = PLANNER_PROMPT | llm | parser
    writer_chain = WRITER_PROMPT | llm | parser

    def parse_plan(text: str) -> Dict[str, Any]:
        """
        Extracts JSON object from LLM output safely.
        Handles cases like:
        - "Sure, here's the JSON: { ... }"
        """
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1:
                raise ValueError("No JSON object found in planner output.")
            json_str = text[start:end + 1]
            return json.loads(json_str)


    # we'll do planner separately in main() for memory saving
    return planner_chain, writer_chain, parse_plan
