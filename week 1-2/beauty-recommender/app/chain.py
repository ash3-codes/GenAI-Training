"""from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException

from app.schema import IngredientResponse
from app.prompt import ingredient_prompt
from app.llm import llm


parser = PydanticOutputParser(pydantic_object=IngredientResponse)
format_instructions = parser.get_format_instructions()


def get_ingredient_report(
    skin_type: str,
    skin_needs: str,
    concern: str,
    product_type: str,
    format_instructions = parser.get_format_instructions(),
):
    # Inject parser format instructions into user prompt
    prompt_with_format = ingredient_prompt.partial(
        format_instructions=parser.get_format_instructions()
    )

    # SINGLE LLM CALL CHAIN
    chain = prompt_with_format | llm | parser

    try:
        result = chain.invoke(
            {
                "skin_type": skin_type or "normal",
                "skin_needs": skin_needs or "general maintenance",
                "concern": concern or "general skin health",
                "product_type": product_type or "serum",
            }
        )
        return {
            "ok": True,
            "data": result.model_dump(),
            "raw_obj": result,
        }

    except OutputParserException as e:
        # Safe fail: return error + raw output for debugging
        return {
            "ok": False,
            "error": "Invalid output format produced by LLM",
            "details": str(e),
        }

    except Exception as e:
        return {
            "ok": False,
            "error": "Unexpected error",
            "details": str(e),
        }
"""

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException

from app.schema import IngredientResponse
from app.prompt import ingredient_prompt
from app.llm import llm

parser = PydanticOutputParser(pydantic_object=IngredientResponse)

def get_ingredient_report(skin_type, skin_needs, concern, product_type):
    chain = ingredient_prompt | llm | parser

    try:
        result = chain.invoke(
            {
                "skin_type": skin_type or "normal",
                "skin_needs": skin_needs or "general maintenance",
                "concern": concern or "general skin health",
                "product_type": product_type or "serum",
                "format_instructions": parser.get_format_instructions(), 
            }
        )

        return {"ok": True, "data": result.model_dump()}

    except OutputParserException as e:
        return {
            "ok": False,
            "error": "Invalid output format produced by LLM",
            "details": str(e),
        }
