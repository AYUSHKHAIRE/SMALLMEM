import streamlit as st
from pre_processing.pdf import PDFProcessor
from RAG.pre_processor import Chunker
from RAG.embeding import Embedder
from LLM.conversation import ConversationChain
from LLM.gemma_local import GemmaManager
import json
from config.logger_config import logger

st.header("Chat with multiple pdfs 📚")

with st.spinner("Loading the app..."):
    # Initialize persistent objects in session state
    if "EB" not in st.session_state:
        st.session_state.EB = Embedder()
    if "PP" not in st.session_state:
        st.session_state.PP = PDFProcessor()
    if "CK" not in st.session_state:
        st.session_state.CK = Chunker()
    if "CC" not in st.session_state:
        st.session_state.CC = ConversationChain()
    if "GM" not in st.session_state:
        st.session_state.GM = GemmaManager()

EB = st.session_state.EB
PP = st.session_state.PP
CK = st.session_state.CK
CC = st.session_state.CC
GM = st.session_state.GM

question = st.text_input(
    "Your question:",
    placeholder="Type your question here...",
    key="question",
    value="The best and most beautiful things in the world cannot be seen or even touched—they must be felt with the heart."
)
ask_button = st.button("Ask", key="ask_button")

with st.sidebar:
    st.subheader("Upload PDF files and click on 'Process'")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    button = st.button("Process", key="process_button")

    if button:
        if uploaded_files:
            st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")
            with st.spinner("Processing files..."):
                full_md_text = []
                for uploaded_file in uploaded_files:
                    file_path = f"uploads/{uploaded_file.name}"
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # extract markdown
                    full_md_text.append(PP.get_markdown(file_path))
                    st.markdown(f"**Extracted Text:** {full_md_text[-1][:500]}...")

                # chunking
                text_chunks = []
                for md_text in full_md_text:
                    text_chunks.extend(CK.chunk_text(md_text))

                # embedding
                try:
                    EB.all_embeddings = EB.embed_chunks(text_chunks)
                    EB.chunks = text_chunks
                except Exception as e:
                    st.error(f"Error during embedding: {e}")

                st.session_state.embeddings_ready = True
                st.success(f"Processed {len(uploaded_files)} file(s) into {len(text_chunks)} chunks.")
        else:
            st.warning("Please upload at least one PDF file.")

# ---- ASK SECTION ----
if ask_button and question:
    if not st.session_state.get("embeddings_ready", False):
        st.warning("Please upload and process PDF files first.")
    else:
        CC.add_message(question, "user")
        with st.spinner("Finding relevant information..."):
            query_embedding = EB.get_user_query_embedding(question)
            top_k_scores, top_k_texts = EB.get_similar_chunks(query_embedding, top_k=10)
            final_context = ""
            for i, chunk in enumerate(top_k_texts):
                final_context += chunk + "\n\n\n"
            prompt = f"""
                You are an AI assistant. Use the following context to answer the question.
                Context: {final_context}
                Question: {question}
                return answer as a markdown string
            """
            answer = GM.ask_query(prompt)
            logger.debug(f"Generated answer: {answer}")
            CC.add_message(answer, "assistant")
        st.subheader("Answer:")
        st.markdown(answer)
        st.balloons()