import os
from datetime import datetime

from RAG.pre_processor import Chunker
from RAG.embeding import Embedder
from Indexer.conccontx import ConcentratedContext
from LLM.docker_model import dockerModel


class SinglePDFEngine:
    def __init__(self, model="ai/granite-4.0-h-tiny:7B", port=12434):
        self.chunker = Chunker()
        self.embedder = Embedder()

        self.llm = dockerModel(
            model=model,
            hostname="localhost",
            port=12434,
            stream=True,
            system_prompt="You are a helpful assistant.",
        )

    # --------------------------------------------------------------
    #  Ask LLM
    # --------------------------------------------------------------
    def ask_LLM(self, question, context=""):
        prompt = f"""
You are a helpful assistant.

Question:
{question}

Context:
{context}

Answer:
        """

        response = self.llm.ask_query(prompt[:18000])

        if hasattr(response, "__iter__") and not isinstance(response, str):
            return "".join(chunk for chunk in response)

        return response or ""

    # --------------------------------------------------------------
    # Build RAG, Index, Hybrid contexts
    # --------------------------------------------------------------
    def build_contexts(self, question, full_document_text, top_k=5):

        chunks = self.chunker.chunk_text(full_document_text)

        # ----------- RAG -------------
        q_emb = self.embedder.get_user_query_embedding(question)
        _, rag_chunks = self.embedder.get_similar_chunks(q_emb, top_k)
        rag_context = "\n\n".join(rag_chunks) if rag_chunks else ""

        # ----------- INDEX / CCX -------------
        ccx = ConcentratedContext("CCX_SINGLE")
        ccx.build(chunks)

        index_context = ccx.generate("", question, k=top_k)
        if index_context == "No concentrated context found.":
            index_context = ""

        # ----------- HYBRID -------------
        if rag_context and index_context:
            hybrid_context = (
                f"Semantic:\n{rag_context}\n\nLiteral:\n{index_context}"
            )
        else:
            hybrid_context = rag_context or index_context

        return rag_context, index_context, hybrid_context

    # --------------------------------------------------------------
    # Evaluate a single question using a single PDF text
    # --------------------------------------------------------------
    def evaluate_single(self, question , pdf_text):
        rag_ctx, index_ctx, hybrid_ctx = self.build_contexts(question, pdf_text)

        modes = {
            "default": "",
            "rag": rag_ctx,
            "index": index_ctx,
            "hybrid": hybrid_ctx,
        }

        results = {}

        for mode, ctx in modes.items():
            start = datetime.now()
            pred = self.ask_LLM(question, ctx)
            latency = (datetime.now() - start).total_seconds()

            results[mode] = {
                "context": ctx,
                "answer": pred,
                "latency": latency,
            }

        return results
