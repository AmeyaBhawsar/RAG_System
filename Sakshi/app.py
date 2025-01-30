import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize LLM and embeddings
llm = ChatGroq(model="llama3-8b-8192")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = InMemoryVectorStore(embeddings)

# Streamlit UI
st.title("PDF-based Question Answering with RAG")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    
    # Load and split PDF
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    
    # Index chunks
    vector_store.add_documents(documents=all_splits)
    st.success("PDF Uploaded and Indexed Successfully!")

question = st.text_input("Ask a question about the PDF:")
if question:
    # Retrieve relevant context
    retrieved_docs = vector_store.similarity_search(question)
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    # Generate answer using Groq LLM
    prompt_template = """
    Answer the question based on the context below. If you can't answer, say "I don't know."
    
    Context: {context}
    
    Question: {question}
    """
    prompt = PromptTemplate.from_template(prompt_template)
    messages = prompt.invoke({"question": question, "context": docs_content})
    response = llm.invoke(messages)
    
    # Display answer
    st.subheader("Answer:")
    st.write(response.content)
