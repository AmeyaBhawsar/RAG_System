import re
from PyPDF2 import PdfReader
from typing import List

class FileHandler:
    def __init__(self, files):
        self.files = files

    def _read_pdf(self, file):
        """Extract text from a PDF file."""
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    def _read_txt(self, file):
        """Extract text from a TXT file."""
        return file.read().decode("utf-8")

    def preprocess_text(self, text):
        """Preprocess text using regex and separators."""
        text = re.sub(r"\s+", " ", text)  # Remove extra spaces
        return text

    def get_documents(self):
        """Extract and preprocess text from uploaded files."""
        documents = []
        for file in self.files:
            if file.name.endswith(".pdf"):
                text = self._read_pdf(file)
            elif file.name.endswith(".txt"):
                text = self._read_txt(file)
            else:
                raise ValueError("Unsupported file format.")
            documents.append(self.preprocess_text(text))
        return documents
    
    