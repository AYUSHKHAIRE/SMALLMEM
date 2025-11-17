import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm

from RAG.pre_processor import Chunker
from RAG.embeding import Embedder
from Indexer.conccontx import ConcentratedContext
from LLM.docker_model import dockerModel


class Engine:
    def __init__(self, model="ai/granite-4.0-h-tiny:7B", port=12434):
        self.chunker = Chunker()
        self.embedder = Embedder()
        self.indexer = ConcentratedContext("CCX")

        self.llm = dockerModel(
            model=model,
            hostname="localhost",
            port=port,
            stream=True,
            system_prompt="You are a helpful assistant."
        )

    # ----------------------------
    # LLM Ask Function
    # ----------------------------
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

    # ----------------------------
    # Build contexts for one item
    # ----------------------------
    def build_contexts(self, question, merged_context, top_k=5):

        # ---- Split Context ----
        chunks = self.chunker.chunk_text(merged_context)

        # ---- RAG ----
        q_emb = self.embedder.get_user_query_embedding(question)
        _, rag_chunks = self.embedder.get_similar_chunks(q_emb, top_k)
        rag_context = "\n\n".join(rag_chunks) if rag_chunks else ""

        # ---- INDEX (CCX) ----
        ccx = ConcentratedContext("CCX_SINGLE")
        ccx.build(chunks)

        index_context = ccx.generate("", question, k=top_k)
        if index_context == "No concentrated context found.":
            index_context = ""

        # ---- Hybrid ----
        if rag_context and index_context:
            hybrid_context = (
                f"Semantic Context:\n{rag_context}\n\n"
                f"Literal Evidence:\n{index_context}"
            )
        else:
            hybrid_context = rag_context or index_context

        return rag_context, index_context, hybrid_context

    # ----------------------------
    # Main Evaluation Method
    # ----------------------------
    def evaluate(self, qa_df, output_file="evaluation_results.csv"):

        rows = []

        for q, a, ctx in tqdm(
            zip(qa_df["question"], qa_df["answer"], qa_df["merged_context"]),
            total=len(qa_df),
            desc="Evaluating dataset"
        ):

            # Build all contexts
            rag_ctx, index_ctx, hybrid_ctx = self.build_contexts(q, ctx)

            results_for_modes = {}

            modes = {
                "default": "",
                "rag": rag_ctx,
                "index": index_ctx,
                "hybrid": hybrid_ctx
            }

            # Run LLM for each mode
            for mode_name, mode_ctx in modes.items():

                start = datetime.now()
                pred = self.ask_LLM(q, mode_ctx)
                latency = (datetime.now() - start).total_seconds()

                results_for_modes[mode_name] = {
                    "context": mode_ctx,
                    "llm_answer": pred,
                    "latency": latency
                }

            # Store row
            rows.append({
                "question": q,
                "gold_answer": a,
                "merged_context": ctx,

                # Default
                "default_context": results_for_modes["default"]["context"],
                "default_llm_answer": results_for_modes["default"]["llm_answer"],
                "default_latency": results_for_modes["default"]["latency"],

                # RAG
                "rag_context": results_for_modes["rag"]["context"],
                "rag_llm_answer": results_for_modes["rag"]["llm_answer"],
                "rag_latency": results_for_modes["rag"]["latency"],

                # Index
                "index_context": results_for_modes["index"]["context"],
                "index_llm_answer": results_for_modes["index"]["llm_answer"],
                "index_latency": results_for_modes["index"]["latency"],

                # Hybrid
                "hybrid_context": results_for_modes["hybrid"]["context"],
                "hybrid_llm_answer": results_for_modes["hybrid"]["llm_answer"],
                "hybrid_latency": results_for_modes["hybrid"]["latency"],
            })

        df = pd.DataFrame(rows)
        os.makedirs("results", exist_ok=True)
        df.to_csv(f"results/{output_file}", index=False)
        return df
