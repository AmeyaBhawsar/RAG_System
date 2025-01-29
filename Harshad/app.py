# Working code

import os
from dotenv import load_dotenv
import streamlit as st
from resume_parser import process_resume
from langchain import hub
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import START, StateGraph
from langchain_core.documents import Document
from typing_extensions import List, TypedDict

load_dotenv()

# Initialize the LLM and other components
llm = ChatGroq(model="llama3-8b-8192", api_key=os.getenv("API_KEY", ""))
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = Chroma(persist_directory="vector_store", embedding_function=embedding_model)

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Streamlit interface
st.title("Resume Parser Chatbot")
st.sidebar.header("Upload your resume")

uploaded_file = st.sidebar.file_uploader("Choose a PDF or Text file", type=["pdf", "txt"])

if uploaded_file is not None:
    # Process the resume (PDF or Text)
    try:
        vector_store, all_splits = process_resume(uploaded_file)
        st.sidebar.success("Resume uploaded and processed. You can now ask questions!")

        # User input for the question
        question = st.text_input("Ask a question about the resume:")

        if question:
            # Initialize state for the pipeline
            state = {"question": question, "context": [], "answer": ""}
            
            # Step 1: Retrieve relevant documents
            retrieved_docs = vector_store.similarity_search(state["question"])
            state["context"] = retrieved_docs

            # Step 2: Generate answer
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            messages = prompt.invoke({"question": state["question"], "context": docs_content})
            response = llm.invoke(messages)
            state["answer"] = response.content

            # Display the answer
            st.write(f"**Answer:** {state['answer']}")
    except ValueError as e:
        st.sidebar.error(str(e))

else:
    st.sidebar.info("Please upload a resume to get started.")
