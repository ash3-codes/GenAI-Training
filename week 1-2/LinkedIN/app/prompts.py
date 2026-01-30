# app/prompts.py
from langchain_core.prompts import ChatPromptTemplate

PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         """You are an expert LinkedIn technical storyteller and post strategist.
You will plan a post for a technical audience (software/AI/ML engineers).
You MUST preserve timeline correctness and only use relevant past topics.

Rules:
- Do NOT invent experiences
- Keep tone professional, not cringe
- Link past topics only if they logically support the current week's work
- Plan must include storyline + structure + relevance explanation
"""),
        ("human",
         """PAST MEMORY (recent weeks):
{weekly_memory}

LONG SUMMARY MEMORY:
{summary_memory}

CURRENT WEEK INPUT:
{this_week_text}

WORD LIMIT:
{word_limit}

Task:
1) Extract key work done this week
2) Decide which past topics are relevant (if any) and why
3) Create a structured plan for the LinkedIn post

Return output strictly in this JSON format:
{{
  "this_week_topics": ["..."],
  "relevant_memory_used": [
    {{
      "week": "....",
      "topic": "...",
      "why_relevant": "..."
    }}
  ],
  "story_arc": "1 sentence",
  "post_outline": ["hook", "work summary", "learnings", "closing"],
  "keywords": ["..."]
}}
"""),
    ]
)

WRITER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         """You are a LinkedIn technical storyteller.
Write posts for technical audience with clarity and credibility.

Rules:
- Must be within word limit
- Must sound human, professional
- Avoid emojis spam, avoid generic motivation lines
- Mention timeline + continuity only if relevant
- Do not hallucinate
"""),
        ("human",
         """PLAN JSON:
{plan_json}

WRITE the final LinkedIn post.

Constraints:
- EXACT word limit: {word_limit} words (±5 words allowed)
- Use simple paragraphs
- End with 3-5 relevant hashtags (technical)
""")
    ]
)
