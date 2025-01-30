import streamlit as st
from llm_handler import LLMHandler
from embedding_handler import EmbeddingHandler, EMBEDDING_OPTIONS
from vector_db_handler import VectorDBHandler, VECTOR_DB_OPTIONS
from file_handler import FileHandler
from reranker import re_rank_cross_encoders

# Streamlit UI
st.set_page_config(layout="wide")
st.title("Custom RAG Pipeline")

# Split screen into two columns
left_col, right_col = st.columns(2)

LLM_OPTIONS = {
    "Mixtral (8x7B)": "mixtral-8x7b-32768",
    "Llama 3 (8B)": "llama3-8b-8192",
    "Llama 3 (70B)": "llama3-70b-8192",
    "Gemma2 (9B)": "gemma2-9b-it",
    "Llama 3.2 (3B)": "llama3.2:3b",  # The model you have installed
}

# Left column: Component selection
with left_col:
    st.header("Component Selection")

    # LLM selection
    llm_model = st.selectbox("Select LLM", list(LLM_OPTIONS.keys()))
    llm_model = LLM_OPTIONS[llm_model]
    llm_handler = LLMHandler(llm_model)

    # Embedding selection
    embedding_type = st.selectbox("Select Embedding Type", list(EMBEDDING_OPTIONS.keys()))
    embedding_model = st.selectbox("Select Embedding Model", list(EMBEDDING_OPTIONS[embedding_type].keys()))
    embedding_handler = EmbeddingHandler(embedding_type, embedding_model)

    # Vector DB selection
    vector_db_type = st.selectbox("Select Vector DB", list(VECTOR_DB_OPTIONS.keys()))
    vector_db_handler = VectorDBHandler(vector_db_type, embedding_handler.get_embeddings())

    # File upload
    uploaded_files = st.file_uploader("Upload PDF or TXT files", accept_multiple_files=True)
    if uploaded_files:
        file_handler = FileHandler(uploaded_files)
        documents = file_handler.get_documents()  # This now returns chunks

    # Integrate button
    if st.button("Integrate Components"):
        st.success("Pipeline integrated successfully!")

# Right column: Q&A bot
with right_col:
    st.header("🗣️ Chat with your PDF...")
    prompt = st.text_area("**Ask a question related to your document:**")
    ask = st.button("🔥 Ask")

    if ask and prompt and uploaded_files:
        # Rerank documents (chunks)
        relevant_text, relevant_text_ids = re_rank_cross_encoders(prompt, documents)
        
        # Call LLM with streaming
        response = llm_handler.call_llm(context=relevant_text, prompt=prompt)
        
        # Display streaming response
        response_container = st.empty()
        full_response = ""
        for chunk in response:
            full_response += chunk
            response_container.markdown(full_response)
        
        # Display retrieved documents (chunks)
        with st.expander("See retrieved documents"):
            st.write(relevant_text)
        
        # Display relevant document IDs (chunk IDs)
        with st.expander("See most relevant document ids"):
            st.write(relevant_text_ids)
