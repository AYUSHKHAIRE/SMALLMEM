import streamlit as st
from pre_processing.pdf import PDFProcessor
from RAG.pre_processor import Chunker
from RAG.embeding import Embedder
from LLM.conversation import ConversationChain
from LLM.docker_model import dockerModel 
from config.logger_config import logger
import os
from Indexer.conccontx import ConcentratedContext

# -----------------------------------------
# INITIAL LOADING
# -----------------------------------------
st.set_page_config(page_title="RAG vs CCX Comparison", layout="wide")

st.title("📘 RAG vs CCX — Side by Side Answer Comparison")

app_progress_text = "Loading components..."
col1, col2 = st.columns([1, 1]) 
with col1:
    app_my_bar = st.progress(0, text=app_progress_text)
with col2:
    LLM_answer_bar = st.progress(0, text="Waiting for input...")

# --- Initialize persistent objects ---
if "EB" not in st.session_state:
    st.session_state.EB = Embedder()
    app_my_bar.progress(20, text="Loading embedder...")

if "PP" not in st.session_state:
    st.session_state.PP = PDFProcessor()
    app_my_bar.progress(40, text="Loading PDF processor...")

if "DM" not in st.session_state:
    st.session_state.DM = dockerModel(
        model="ai/granite-4.0-h-tiny:7B",
        hostname="localhost",
        port=12434,
        stream=True,
        system_prompt="You are a helpful assistant."
    )
    app_my_bar.progress(55, text="Loading LLM (Docker)...")

if "CK" not in st.session_state:
    st.session_state.CK = Chunker()
    app_my_bar.progress(70, text="Loading chunker...")

if "CC" not in st.session_state:
    st.session_state.CC = ConversationChain()
    app_my_bar.progress(85, text="Loading conversation chain...")

if "CCX" not in st.session_state:
    st.session_state.CCX = ConcentratedContext(name="collection")
    app_my_bar.progress(95, text="Loading concentrated context...")

if "generating" not in st.session_state:
    st.session_state.generating = False

if "embeddings_ready" not in st.session_state:
    st.session_state.embeddings_ready = False

app_my_bar.progress(100, text="App ready!")

EB = st.session_state.EB
PP = st.session_state.PP
CK = st.session_state.CK
CC = st.session_state.CC
DM = st.session_state.DM
CCX = st.session_state.CCX

# -----------------------------------------
# SIDEBAR - PDF UPLOAD
# -----------------------------------------
with st.sidebar:
    st.subheader("📄 Upload PDF files")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    button = st.button("Process PDFs")

    pdf_processor_bar = st.progress(0, text="Waiting...")

    if button:
        if uploaded_files:
            pdf_processor_bar.progress(25, text="Reading PDFs...")
            full_md_text = []

            for uploaded_file in uploaded_files:
                file_path = f"uploads/{uploaded_file.name}"
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                md_text = PP.get_markdown(file_path)
                full_md_text.append(md_text)

                with st.expander(f"Preview - {uploaded_file.name}"):
                    st.markdown(md_text[:500] + "...")

            pdf_processor_bar.progress(50, text="Chunking...")
            text_chunks = []
            for md_text in full_md_text:
                text_chunks.extend(CK.chunk_text(md_text))

            pdf_processor_bar.progress(70, text="Embedding chunks...")
            try:
                EB.all_embeddings = EB.embed_chunks(text_chunks)
                EB.chunks = text_chunks

                pdf_processor_bar.progress(85, text="Building CCX index...")
                CCX.build(text_chunks)

                pdf_processor_bar.progress(100, text="Processing completed!")
                st.session_state.embeddings_ready = True

                st.success(f"Processed {len(text_chunks)} chunks.")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Upload at least one PDF file!")

# -----------------------------------------
# CHAT SECTION
# -----------------------------------------
st.subheader("💬 Ask a question about your documents")

messages = st.session_state.CC.messages
for msg in messages:
    with st.chat_message(msg["sender"]):
        st.markdown(msg["message"])
        st.caption(msg["timestamp"])

question = st.chat_input("Type a question...")

# -----------------------------------------
# ANSWER GENERATION
# -----------------------------------------
if question and not st.session_state.generating:

    if not st.session_state.embeddings_ready:
        st.warning("Please upload & process PDFs first.")
        st.stop()

    st.session_state.generating = True
    st.session_state.CC.add_message(question, "user")

    chat_history = "" # temprory

    with st.chat_message("user"):
        st.markdown(question)

    # --- Retrieval Phase ---
    LLM_answer_bar.progress(20, text="Embedding query...")
    query_embedding = EB.get_user_query_embedding(question)

    LLM_answer_bar.progress(40, text="Retrieving RAG chunks...")
    _, rag_chunks = EB.get_similar_chunks(query_embedding, top_k=5)
    rag_context = "\n\n".join(rag_chunks)

    LLM_answer_bar.progress(55, text="Retrieving CCX chunks...")
    ccx_context = CCX.generate("", question, k=5)

    # --- Build prompts ---
    base_prompt = """
You are a helpful AI assistant. Use the context below when appropriate.

Rules:
- If context helps, use it.
- If unrelated, ignore it.
- Keep explanations short & clean.

Question: {question}

Context:
{context}

Answer:
"""

    rag_prompt = base_prompt.format(question=question, context=rag_context, history="")
    ccx_prompt = base_prompt.format(question=question, context=ccx_context, history="")
    hybrid_context = f"Semantic Context (RAG):\n{rag_context}\n\nLiteral Evidence (CCX):\n{ccx_context}"
    hybrid_prompt = base_prompt.format(question=question, context=hybrid_context, history="")

    # logger.debug(f"{hybrid_context}")

    col_rag, col_ccx, col_hybrid = st.columns(3)

    with col_rag:
        st.markdown("### 🔵 RAG (Embedding-Based Answer)")
        rag_placeholder = st.empty()
        rag_text = ""

    with col_ccx:
        st.markdown("### 🟠 CCX (Keyword Index Answer)")
        ccx_placeholder = st.empty()
        ccx_text = ""

    with col_hybrid:
        st.markdown("### 🟣 Hybrid (RAG + CCX)")
        hybrid_placeholder = st.empty()
        hybrid_text = ""


    # --- Generate both answers ---
    LLM_answer_bar.progress(70, text="Generating RAG answer...")
    for chunk in DM.ask_query(rag_prompt[:200000]):
        rag_text += chunk
        rag_placeholder.markdown(rag_text)

    LLM_answer_bar.progress(80, text="Generating CCX answer...")
    for chunk in DM.ask_query(ccx_prompt[:200000]):
        ccx_text += chunk
        ccx_placeholder.markdown(ccx_text)

    LLM_answer_bar.progress(90, text="Generating Hybrid answer...")
    for chunk in DM.ask_query(hybrid_prompt[:200000]):
        hybrid_text += chunk
        hybrid_placeholder.markdown(hybrid_text)

    LLM_answer_bar.progress(100, text="Done!")

    st.session_state.CC.add_message(
    f"**RAG Answer:**\n{rag_text}\n\n**CCX Answer:**\n{ccx_text}\n\n**Hybrid Answer:**\n{hybrid_text}",
    "assistant"
    )


    st.session_state.generating = False

else:
    st.info("Upload PDFs and ask something!")
