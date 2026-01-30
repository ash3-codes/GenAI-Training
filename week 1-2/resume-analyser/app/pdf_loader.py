import os
from langchain_community.document_loaders import PyPDFLoader

def load_resume_text(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Resume not found: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()  # list[Document]
    text = "\n".join([d.page_content for d in docs])
    return text
