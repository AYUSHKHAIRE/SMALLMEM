import streamlit as st
from pre_processing.pdf import PDFProcessor
from RAG.pre_processor import Chunker
from RAG.embeding import Embedder
from LLM.conversation import ConversationChain
from LLM.docker_model import dockerModel 
from config.logger_config import logger
import os
# -----------------------------------------
# INITIAL LOADING
# -----------------------------------------
st.set_page_config(page_title="RAG Assistant", layout="wide")

app_progress_text = "Loading the app..."
col1, col2 = st.columns([1, 1]) 
with col1:
    app_my_bar = st.progress(0, text=app_progress_text)
with col2:
    LLM_answer_bar = st.progress(0, text="Waiting for question...")

# --- Initialize persistent objects ---
if "EB" not in st.session_state:
    st.session_state.EB = Embedder()
    app_my_bar.progress(20, text="Loading embedder...")

if "PP" not in st.session_state:
    st.session_state.PP = PDFProcessor()
    app_my_bar.progress(40, text="Loading PDF processor...")

if "CK" not in st.session_state:
    st.session_state.CK = Chunker()
    app_my_bar.progress(60, text="Loading text chunker...")

if "CC" not in st.session_state:
    st.session_state.CC = ConversationChain()
    app_my_bar.progress(80, text="Loading conversation chain...")

if "DM" not in st.session_state:
    st.session_state.DM = dockerModel(
        model="ai/granite-4.0-h-tiny:7B",
        hostname="localhost",
        port=12434,
        stream=True,
        system_prompt="You are a helpful assistant."
    )
    app_my_bar.progress(100, text="App loaded successfully!")

if "generating" not in st.session_state:
    st.session_state.generating = False

if "embeddings_ready" not in st.session_state:
    st.session_state.embeddings_ready = False

app_my_bar.progress(100, text="App loaded successfully!")

EB = st.session_state.EB
PP = st.session_state.PP
CK = st.session_state.CK
CC = st.session_state.CC
DM = st.session_state.DM

# -----------------------------------------
# SIDEBAR - PDF UPLOAD
# -----------------------------------------
with st.sidebar:
    st.subheader("Upload PDF files and click on 'Process'")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    button = st.button("Process", key="process_button")

    pdf_processor_bar = st.progress(0, text="Waiting for files to process...")

    if button:
        if uploaded_files:
            pdf_processor_bar.progress(25, text="Files uploaded. Starting processing...")
            full_md_text = []

            for uploaded_file in uploaded_files:
                file_path = f"uploads/{uploaded_file.name}"
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                md_text = PP.get_markdown(file_path)
                full_md_text.append(md_text)
                with st.expander(f"Preview extracted text ({uploaded_file.name})"):
                    st.markdown(md_text[:500] + "...")

            pdf_processor_bar.progress(50, text="Text extracted. Chunking...")
            text_chunks = []
            for md_text in full_md_text:
                text_chunks.extend(CK.chunk_text(md_text))

            pdf_processor_bar.progress(75, text="Embedding...")
            try:
                EB.all_embeddings = EB.embed_chunks(text_chunks)
                EB.chunks = text_chunks
                pdf_processor_bar.progress(100, text="Embedding completed.")
                st.session_state.embeddings_ready = True
                st.success(f"Processed {len(uploaded_files)} file(s) into {len(text_chunks)} chunks.")
            except Exception as e:
                st.error(f"Error during embedding: {e}")
        else:
            st.warning("Please upload at least one PDF file.")

# -----------------------------------------
# CHAT SECTION
# -----------------------------------------

app_my_bar = st.progress(0, text=app_progress_text)

if uploaded_files:
    tabs = st.tabs([f.name for f in uploaded_files])
    for i, (tab, uploaded_file) in enumerate(zip(tabs, uploaded_files)):
        with tab:
            file_path = os.path.join("uploads", uploaded_file.name)
            st.pdf(file_path)

st.subheader("💬 Chat with your PDF")

# Retrieve all previous messages (preserving order)
messages = st.session_state.CC.messages

# Display previous messages in chronological order
for msg in messages:
    with st.chat_message(msg["sender"]):
        st.markdown(msg["message"])
        st.caption(msg["timestamp"])

# --- Input box at the bottom ---
question = st.chat_input("Ask a question about your documents...")

if question and not st.session_state.generating:
    if not st.session_state.embeddings_ready:
        st.warning("Please upload and process PDF files first.")
    else:
        st.session_state.generating = True
        st.session_state.CC.add_message(question, "user")

        chat_history = st.session_state.CC.get_formatted_context()

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            answer_text = ""

            LLM_answer_bar.progress(0, text="Embedding question...")
            query_embedding = EB.get_user_query_embedding(question)

            LLM_answer_bar.progress(30, text="Retrieving relevant chunks...")
            top_k_scores, top_k_texts = EB.get_similar_chunks(query_embedding, top_k=20)
            final_context = "\n\n\n".join(top_k_texts)

            LLM_answer_bar.progress(60, text="Generating response...")
            prompt = f"""
                    You are a friendly and intelligent AI assistant.

                    You are given:
                    - A user question.
                    - Optional context extracted from uploaded documents.

                    ---

                    ### Guidelines:
                    1. If the question is conversational (like greetings, feelings, or casual chat), respond naturally and briefly as a person would.
                    - Example: If asked "How are you?", reply like "I'm good! How about you?" instead of analyzing the context.
                    2. If the question clearly relates to the provided context, use the context to give a correct and concise answer.
                    3. If the question is unrelated to the context, ignore the context and answer normally using general knowledge.
                    4. Keep your tone warm, simple, and human-like.

                    ---

                    **Question:** {question}

                    **Context (optional):**
                    {final_context}

                    History until now
                    {chat_history}                    
                    ---

                    **Answer:**
                    """

            for chunk in DM.ask_query(prompt):
                answer_text += chunk
                response_placeholder.markdown(answer_text)

            LLM_answer_bar.progress(100, text="Answer generated successfully!")
            st.session_state.CC.add_message(answer_text, "assistant")
            st.session_state.generating = False

else:
    st.info("Upload PDFs to preview them in tabs.")