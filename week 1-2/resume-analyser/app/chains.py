from langchain_core.runnables import RunnableLambda
from llm import llm
from prompts import RESUME_EXTRACT_PROMPT, ATS_PROMPT, JD_EXTRACT_PROMPT, JD_MATCH_PROMPT
from schemas import ResumeExtract, JDExtract, JDMatchReport

def clean_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())

resume_extractor = (
    RunnableLambda(lambda x: {"resume_text": clean_text(x["resume_text"])})
    | RESUME_EXTRACT_PROMPT
    | llm.with_structured_output(ResumeExtract)  # <- key part
)

ats_checker = (
    RunnableLambda(lambda x: {"resume_text": clean_text(x["resume_text"])})
    | ATS_PROMPT
    | llm
)


jd_extractor = (
    RunnableLambda(lambda x: {"jd_text": clean_text(x["jd_text"])})
    | JD_EXTRACT_PROMPT
    | llm.with_structured_output(JDExtract)
)

jd_matcher = (
    JD_MATCH_PROMPT
    | llm.with_structured_output(JDMatchReport)
)
