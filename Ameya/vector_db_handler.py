from langchain_core.vectorstores import InMemoryVectorStore
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS

class VectorDBHandler:
    def __init__(self, db_type, embeddings):
        self.db_type = db_type
        self.embeddings = embeddings

    def get_vector_store(self):
        """Initialize and return the selected vector store."""
        if self.db_type == "In-memory":
            return InMemoryVectorStore(embeddings=self.embeddings)
        elif self.db_type == "Chroma":
            return Chroma(embedding_function=self.embeddings)
        elif self.db_type == "FAISS":
            return FAISS(embedding_function=self.embeddings)
        else:
            raise ValueError("Invalid vector database type.")

# Available vector database options
VECTOR_DB_OPTIONS = {
    "In-memory": "In-memory",
    "Chroma": "Chroma",
    "FAISS": "FAISS",
}

