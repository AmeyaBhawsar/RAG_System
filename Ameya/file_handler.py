import re
import tempfile
import os
from PyPDF2 import PdfReader
from typing import List
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class FileHandler:
    def __init__(self, files):
        self.files = files

    def _read_pdf(self, file):
        """
        Load and split a PDF file into chunks using PyPDFLoader and RecursiveCharacterTextSplitter.
        """
        # Store uploaded file as a temp file
        temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
        temp_file.write(file.read())
        temp_file.close()  # Close the file to ensure it's written to disk

        # Load the PDF using PyPDFLoader
        loader = PyPDFLoader(temp_file.name)
        docs = loader.load()


        # Delete the temp file after loading
        os.unlink(temp_file.name)

        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,  # Adjust chunk size as needed
            chunk_overlap=100,  # Adjust overlap as needed
            separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        )
        chunks = text_splitter.split_documents(docs)

        # Extract text from chunks
        chunk_texts = [chunk.page_content for chunk in chunks]
        return chunk_texts

    def _read_txt(self, file):
        """Extract text from a TXT file."""
        return file.read().decode("utf-8")

    def preprocess_text(self, text):
        """Preprocess text using regex and separators."""
        text = re.sub(r"\s+", " ", text)  # Remove extra spaces
        return text

    def get_documents(self):
        """Extract, preprocess, and chunk text from uploaded files."""
        documents = []
        for file in self.files:
            if file.name.endswith(".pdf"):
                # Use PyPDFLoader and RecursiveCharacterTextSplitter for PDFs
                chunks = self._read_pdf(file)
                documents.extend(chunks)
            elif file.name.endswith(".txt"):
                # Handle TXT files
                text = self._read_txt(file)
                text = self.preprocess_text(text)
                documents.append(text)
            else:
                raise ValueError("Unsupported file format.")
        
        return documents
