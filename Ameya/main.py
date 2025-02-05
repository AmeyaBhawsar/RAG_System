import os
import re
import streamlit as st

from llm_handler import LLMHandler, OllamaLLM
from embedding_handler import EmbeddingHandler, EMBEDDING_OPTIONS
from vector_db_handler import VectorDBHandler, VECTOR_DB_OPTIONS
from file_handler import FileHandler
from reranker import CrossEncoderReranker
from chat_history_handler import ChatHistoryHandler
from langchain_groq import ChatGroq

def is_local_model(model_name: str) -> bool:
    local_models = {"llama3.2:3b"} 
    return model_name in local_models

st.set_page_config(layout="wide")
st.title("Custom RAG Pipeline")

if "chat_history_handler" not in st.session_state:
    st.session_state.chat_history_handler = None  


left_col, right_col = st.columns(2)

LLM_OPTIONS = {
    "Mixtral (8x7B)": "mixtral-8x7b-32768",
    "Llama 3 (8B)": "llama3-8b-8192",
    "Llama 3 (70B)": "llama3-70b-8192",
    "Gemma2 (9B)": "gemma2-9b-it",
    "Llama 3.2 (3B)": "llama3.2:3b",  # Local model via Ollama
}

with left_col:
    st.header("Component Selection")
    
    # LLM selection
    selected_llm_label = st.selectbox("Select LLM", list(LLM_OPTIONS.keys()))
    llm_model = LLM_OPTIONS[selected_llm_label]
    llm_handler = LLMHandler(llm_model)
    
    # Embedding selection
    embedding_type = st.selectbox("Select Embedding Type", list(EMBEDDING_OPTIONS.keys()))
    embedding_model = st.selectbox("Select Embedding Model", list(EMBEDDING_OPTIONS[embedding_type].keys()))
    embedding_handler = EmbeddingHandler(embedding_type, embedding_model)
    st.session_state.embedder = embedding_handler.get_embeddings()
    
    # Vector DB selection
    vector_db_type = st.selectbox("Select Vector DB", list(VECTOR_DB_OPTIONS.keys()))
    vector_db_handler = VectorDBHandler(vector_db_type, embedding_handler.get_embeddings())
    
    # File upload
    uploaded_files = st.file_uploader("Upload PDF or TXT files", accept_multiple_files=True)
    if uploaded_files:
        file_handler = FileHandler(uploaded_files)
        documents = file_handler.get_documents()  
    
    st.session_state.rerank = CrossEncoderReranker()
    
    if st.button("Integrate Components"):
        st.success("Pipeline integrated successfully!")

with right_col:
    st.header("🗣️ Chat with your Document...")
    prompt = st.text_area("**Ask a question or give instructions:**")
    ask = st.button("🔥 Send")


    if st.session_state.chat_history_handler is None:
        if is_local_model(llm_model):
            chat_llm = OllamaLLM(model=llm_model)
        else:
            chat_llm = ChatGroq(model_name=llm_model, api_key=os.getenv("GROQ_API_KEY"))
        st.session_state.chat_history_handler = ChatHistoryHandler(llm=chat_llm)

    chat_handler = st.session_state.chat_history_handler  

    if ask and prompt:
        # Handle name change (or name query) command
        name_pattern = r"(?:your name is|call yourself|you shall be known as) ['\"]?(.+?)['\"]?[\.\?]?$"
        name_match = re.search(name_pattern, prompt, re.IGNORECASE)
        
        if name_match:
            new_name = name_match.group(1)
            chat_handler.set_name(new_name)
            response = f"Got it! My new name is **{new_name}**."

            try:
                chat_handler.add_message("human", prompt)
                chat_handler.add_message("ai", response)
                
            except AttributeError:
                chat_handler.memory.chat_memory.add_user_message(prompt)
                chat_handler.memory.chat_memory.add_ai_message(response)
        else:
            response = chat_handler.chat(prompt)

        response_container = st.empty()
        response_container.markdown(response)

    st.header("📜 Chat History")

    chat_messages = chat_handler.get_chat_history()

    if chat_messages:

        for i in range(0, len(chat_messages), 2):
            if i + 1 < len(chat_messages):
                question = chat_messages[i].content
                answer = chat_messages[i+1].content
                st.markdown(f"**Q:** {question}")
                st.markdown(f"**A:** {answer}")
                st.markdown("---")
            else:
                st.markdown(f"**Message:** {chat_messages[i].content}")
    else:
        st.write("No chat history yet.")


