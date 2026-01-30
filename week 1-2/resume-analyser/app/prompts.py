from langchain_core.prompts import ChatPromptTemplate

RESUME_EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a resume analyser. Extract information STRICTLY from the resume text provided. "
     "Return structured output only."),
    ("human",
     "Resume Text:\n{resume_text}")
])

ATS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an ATS expert. Evaluate ATS friendliness using a strict rubric."),
    ("human",
     """
Check this resume for ATS compatibility.
Return:
- ats_score (0-100)
- issues (list)
- fixes (list)
Resume:\n{resume_text}
""")
])


JD_EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert recruiter. Extract structured information from the Job Description."
     "Return output only in the required schema."),
    ("human", "Job Description:\n{jd_text}")
])

JD_MATCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a technical recruiter and resume evaluator."
     "Compare the resume extraction with the job description extraction."
     "Be strict and realistic. No hallucination. Only use given data."
    ),
    ("human",
     """
Resume Extracted JSON:
{resume_json}

Job Description Extracted JSON:
{jd_json}

Return JD match report strictly as structured output.
""")
])
