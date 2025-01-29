# Working Code

import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Tuple

def process_resume(uploaded_file) -> Tuple[Chroma, List]:
    """
    Process the uploaded resume file (PDF or text) and return the vector store
    and the list of split documents.
    """
    # Create a temporary file to store the uploaded content
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    # Choose loader based on file type
    if uploaded_file.type == "application/pdf":
        loader = PyPDFLoader(temp_file_path)
    elif uploaded_file.type == "text/plain":
        loader = TextLoader(temp_file_path)
    else:
        raise ValueError("Unsupported file format. Please upload a PDF or text file.")
    
    # Load and process documents
    docs = loader.load()
    
    # Split the document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_splits = text_splitter.split_documents(docs)
    
    # Initialize embedding model and vector store
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = Chroma(collection_name="resume_db", embedding_function=embedding_model)
    
    # Add documents to the vector store
    vector_store.add_documents(all_splits)
    
    return vector_store, all_splits
