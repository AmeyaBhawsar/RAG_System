from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings

class EmbeddingHandler:
    def __init__(self, model_type, model_name):
        self.model_type = model_type
        self.model_name = model_name

    def get_embeddings(self):
        """Initialize and return the selected embedding model."""
        if self.model_type == "Hugging Face":
            return HuggingFaceEmbeddings(model_name=self.model_name)
        elif self.model_type == "Ollama":
            return OllamaEmbeddings(model=self.model_name)
        else:
            raise ValueError("Invalid embedding model type.")

# Available embedding options
EMBEDDING_OPTIONS = {
    "Hugging Face": {
        "sentence-transformers/all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-mpnet-base-v2": "sentence-transformers/paraphrase-mpnet-base-v2",
    },
    "Ollama": {
        "llama3": "llama3",
        "mxbai-embed-large": "mxbai-embed-large",
        "nomic-embed-text": "nomic-embed-text",
    },
}
