import streamlit as st
from pre_processing.pdf import PDFProcessor
from RAG.pre_processor import Chunker
from RAG.embeding import Embedder
from LLM.conversation import ConversationChain
from LLM.gemma_local import GemmaManager
import json
from config.logger_config import logger

app_progress_text =  "Loading the app..."

col1, col2 = st.columns([1, 1]) 
with col1:
    app_my_bar = st.progress(0, text=app_progress_text)
with col2:
    LLM_answer_bar = st.progress(0, text="Waiting for question...")

# Initialize persistent objects in session state
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
if "GM" not in st.session_state:
    st.session_state.GM = GemmaManager()
    app_my_bar.progress(100, text="App loaded successfully!")

app_my_bar.progress(100, text="App loaded successfully!")

app_progress_text =  "App loaded successfully!"

EB = st.session_state.EB
PP = st.session_state.PP
CK = st.session_state.CK
CC = st.session_state.CC
GM = st.session_state.GM

col_3, col_4 = st.columns([6, 1])
with col_3:
    question = st.text_input(
        label="Your question:",
        placeholder="Type your question here...",
        key="question",
        value="explain about his projects",
        label_visibility="collapsed"
    )
with col_4:
    ask_button = st.button("🤔", key="ask_button")

with st.sidebar:
    st.subheader("Upload PDF files and click on 'Process'")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    button = st.button("Process", key="process_button")

    pdf_processor_bar = st.progress(0, text="Waiting for files to process...")

    if button:
        if uploaded_files:
            pdf_processor_text = "Processing files..."
            pdf_processor_bar.progress(0, text=pdf_processor_text)
            pdf_processor_bar.progress(25, text="Files uploaded. Starting processing...")
            full_md_text = []
            for uploaded_file in uploaded_files:
                file_path = f"uploads/{uploaded_file.name}"
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # extract markdown
                full_md_text.append(PP.get_markdown(file_path))
                with st.expander(f"Preview extracted markdown from {uploaded_file.name}"):
                    st.markdown(f"**Extracted Text:** {full_md_text[-1][:500]}...")
            pdf_processor_bar.progress(50, text=f"Extracted text from {uploaded_file.name}")

            # chunking
            text_chunks = []
            for md_text in full_md_text:
                text_chunks.extend(CK.chunk_text(md_text))
            pdf_processor_bar.progress(75, text="Text chunking completed.")

            # embedding
            try:
                EB.all_embeddings = EB.embed_chunks(text_chunks)
                EB.chunks = text_chunks
                pdf_processor_bar.progress(100, text="Embedding completed.")
            except Exception as e:
                st.error(f"Error during embedding: {e}")

            st.session_state.embeddings_ready = True
            st.success(f"Processed {len(uploaded_files)} file(s) into {len(text_chunks)} chunks.")
            pdf_processor_text = "Processing completed successfully!"
        else:
            st.warning("Please upload at least one PDF file.")

# ---- ASK SECTION ----

if ask_button and question:
    if not st.session_state.get("embeddings_ready", False):
        st.warning("Please upload and process PDF files first.")
    else:
        # === Step 1: Progress setup ===
        LLM_answer_bar.progress(20, text="Finding relevant information...")
        CC.add_message(question, "user")

        # === Step 2: Compute query embedding ===
        LLM_answer_bar.progress(40, text="Preparing user embedding...")
        query_embedding = EB.get_user_query_embedding(question)

        # === Step 3: Retrieve top-k chunks ===
        LLM_answer_bar.progress(60, text="Retrieving relevant chunks...")
        top_k_scores, top_k_texts = EB.get_similar_chunks(query_embedding, top_k=5)

        # === Step 4: Build context ===
        final_context = "\n\n\n".join(top_k_texts)
        prompt = f"""
        You are an AI assistant. Use the following context to answer the question.
        Question: {question}
        Context: {final_context}
        Provide a detailed and accurate answer based on the context.
        """

        # === Step 5: Start generating ===
        LLM_answer_bar.progress(80, text="Generating answer...")

        # Stream output dynamically
        st.subheader("Answer:")
        response_placeholder = st.empty()
        answer_text = ""

        for chunk in GM.ask_query(prompt):
            answer_text += chunk
            response_placeholder.markdown(answer_text)

        # === Step 6: Wrap up ===
        LLM_answer_bar.progress(100, text="Answer generated successfully!")
        logger.debug(f"Generated answer: {answer_text}")
        CC.add_message(answer_text, "assistant")
        st.balloons()
