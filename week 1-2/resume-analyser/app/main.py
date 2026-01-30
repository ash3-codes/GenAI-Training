import json
from pdf_loader import load_resume_text
from chains import resume_extractor, ats_checker, jd_extractor, jd_matcher

RESUME_PATH = "resume_data/resume.pdf"

def main():
    resume_text = load_resume_text(RESUME_PATH)

    print("\nPaste the Job Description (end input with ENTER + CTRL+Z on Windows):\n")
    jd_text = ""
    while True:
        try:
            line = input()
            jd_text += line + "\n"
        except EOFError:
            break

    extracted_resume = resume_extractor.invoke({"resume_text": resume_text})
    extracted_jd = jd_extractor.invoke({"jd_text": jd_text})

    match_report = jd_matcher.invoke({
        "resume_json": extracted_resume.model_dump_json(indent=2),
        "jd_json": extracted_jd.model_dump_json(indent=2),
    })

    ats_report = ats_checker.invoke({"resume_text": resume_text})

    print("\n===== RESUME EXTRACT =====\n")
    print(extracted_resume.model_dump_json(indent=2))

    print("\n===== JD EXTRACT =====\n")
    print(extracted_jd.model_dump_json(indent=2))

    print("\n===== JD MATCH REPORT =====\n")
    print(match_report.model_dump_json(indent=2))

    print("\n===== ATS REPORT =====\n")
    print(ats_report.content)

if __name__ == "__main__":
    main()
