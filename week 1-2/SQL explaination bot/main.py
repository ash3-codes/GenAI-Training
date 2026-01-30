import os
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


load_dotenv()


# Deterministic schema

class SQLExplanation(BaseModel):
    natural_language_explaination: str = Field(..., description="return a natural language explaination of what the query does")
    intent: str = Field(..., description="1-2 line deterministic explaination of what query does")

    tables: List[str] = Field(default_factory=list, description="List of all tables referenced")
    joins: List[str] = Field(default_factory=list, description="Join details in simple text")
    selected_fields: List[str] = Field(default_factory=list, description="Columns/expressions selected")

    filters: List[str] = Field(default_factory=list, description="WHERE clause conditions")
    grouping: List[str] = Field(default_factory=list, description="GROUP BY columns, if any")
    having: List[str] = Field(default_factory=list, description="HAVING clause conditions, if any")

    ordering: List[str] = Field(default_factory=list, description="ORDER BY fields, if any")
    limit: Optional[str] = Field(default=None, description="LIMIT/OFFSET, if any")

    notes: List[str] = Field(default_factory=list, description="Any extra caveats or assumptions")


parser = PydanticOutputParser(pydantic_object=SQLExplanation)


# system mssg 

system_msg = """
You are an expert SQL analyst.

Task:
Given a SQL query, explain it deterministically and extract structured details.

Rules:
- Do NOT execute SQL.
- Do NOT assume database engine unless clear.
- Output must strictly follow the JSON schema.
- Do NOT add extra keys.
- No chain-of-thought or reasoning text.
- Be deterministic: same query => same structure & wording style.
""".strip()


# Few shot


fewshot_1_sql = """
SELECT e.name, d.department_name
FROM employees e
JOIN departments d ON e.department_id = d.id
WHERE e.salary > 50000
ORDER BY e.name ASC
LIMIT 10;
""".strip()

fewshot_1_out = """
{
  "natural_language_explaination": "This query finds employees who earn more than 50,000, and for each such employee it returns their name along with the name of the department they belong to. The results are sorted alphabetically by employee name in ascending order, and only the first 10 employees from this sorted list are shown.",
  "intent": "Fetch employee names with their department names for employees earning more than 50000, sorted by name, limited to 10 rows.",
  "tables": ["employees", "departments"],
  "joins": ["employees e JOIN departments d ON e.department_id = d.id"],
  "selected_fields": ["e.name", "d.department_name"],
  "filters": ["e.salary > 50000"],
  "grouping": [],
  "having": [],
  "ordering": ["e.name ASC"],
  "limit": "LIMIT 10",
  "notes": []
}
""".strip()


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_msg),
        ("human", "Here is an example SQL query:\n{example_sql}\n\nExpected output:\n{example_out}"),
        ("human", "Now explain this SQL query:\n{sql_query}\n\n{format_instructions}"),
    ]
).partial(
    example_sql=fewshot_1_sql,
    example_out=fewshot_1_out,
    format_instructions=parser.get_format_instructions(),
)


# llm

llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    temperature=0,
    max_tokens=1024,
)


# chains

chain = prompt | llm | parser


def explain_sql(sql: str) -> SQLExplanation:
    return chain.invoke({"sql_query": sql})


if __name__ == "__main__":

    q = "SELECT r.rd_owner_id, r.rd_owner_name, COALESCE(SUM(s.sales_volume), 0) AS total_sales_volume, COUNT(DISTINCT pi.ingredient_id) AS distinct_ingredient_count FROM rd_owners r JOIN products p ON p.rd_owner_id = r.rd_owner_id LEFT JOIN sales s ON s.product_id = p.product_id LEFT JOIN product_ingredients pi ON pi.product_id = p.product_id GROUP BY r.rd_owner_id, r.rd_owner_name ORDER BY r.rd_owner_id;"


    result = explain_sql(q)
    print("\n--- Explanation (Deterministic JSON) ---")
    print(result.model_dump_json(indent=2))



    """
    Expected output when given input:
    
sql query: 

SELECT
    r.rd_owner_id,
    r.rd_owner_name,
    COALESCE(SUM(s.sales_volume), 0) AS total_sales_volume,
    COUNT(DISTINCT pi.ingredient_id) AS distinct_ingredient_count
FROM rd_owners r
JOIN products p
    ON p.rd_owner_id = r.rd_owner_id
LEFT JOIN sales s
    ON s.product_id = p.product_id
LEFT JOIN product_ingredients pi
    ON pi.product_id = p.product_id
GROUP BY
    r.rd_owner_id,
    r.rd_owner_name
ORDER BY
    r.rd_owner_id;

    
natural language:
For each R&D owner, what is the total sales volume of products they own, and how many distinct ingredients are used across those products?

"""
