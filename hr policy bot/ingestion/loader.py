import pdfplumber
import os

class PDFLoader:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def load_pdfs(self):
        documents = []

        for file in os.listdir(self.folder_path):
            if file.endswith(".pdf"):
                full_path = os.path.join(self.folder_path, file)
                documents.append(self._extract_pdf(full_path))

        return documents

    def _extract_pdf(self, file_path):
        pages_data = []

        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                pages_data.append({
                    "page_number": i + 1,
                    "text": page.extract_text(),
                    "tables": page.extract_tables(),
                })

        return {
            "doc_name": os.path.basename(file_path),
            "pages": pages_data
        }