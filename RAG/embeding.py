from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Optional
from config.logger_config import logger
from RAG.vector import QdrantClientWrapper

# ---------------------------
# Top-level worker (pickleable)
# ---------------------------
def _mean_pooling_from_output(model_output, attention_mask):
    token_embeddings = model_output[0]  # (batch, seq_len, hidden)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    denom = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / denom

def embed_chunk_group_worker(args):
    """
    Top-level worker used by ProcessPoolExecutor.
    args: tuple(model_name, chunk_group, device_str)
    Each worker loads its own tokenizer & model.
    Returns: list of lists (embedding vectors)
    """
    model_name, chunk_group, device_str = args
    device = torch.device(device_str)
    # Load model & tokenizer inside worker
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    out_vectors = []
    for text in chunk_group:
        encoded = tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            model_output = model(**encoded)
        emb = _mean_pooling_from_output(model_output, encoded["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)  # shape (1, dim)
        out_vectors.append(emb.squeeze(0).cpu().numpy().tolist())
    return out_vectors

# ---------------------------
# Embedder class
# ---------------------------
class Embedder:
    def __init__(
        self,
        collection_name: str = "collection_1",
        model_name: str = "BAAI/bge-m3",
        device: Optional[torch.device] = None,
    ):
        self.model_name = model_name
        self.collection_name = collection_name
        self.device = device or torch.device("cpu")  # default CPU; change to cuda if needed
        self.tokenizer = None
        self.model = None
        self.all_embeddings: List[np.ndarray] = []
        self.chunks: List[str] = []
        self.qdrant_enabled = False
        self.qdrant = None

        # Try Qdrant
        try:
            self.qdrant = QdrantClientWrapper(url="http://localhost:6333")
            self.qdrant_enabled = True
            logger.info(f"Qdrant connected. Using collection: {self.collection_name}")
        except Exception as e:
            self.qdrant = None
            self.qdrant_enabled = False
            logger.warning(f"Qdrant not available ({e}). Falling back to in-memory search.")

        logger.info(f"Embedder init: model={self.model_name}, device={self.device}")
        self.load_model()

    def load_model(self):
        """Load tokenizer & model into this process (for threaded/shared usage)."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.model is None:
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
        logger.info(f"Loaded model/tokenizer: {self.model_name} on {self.device}")

    def _embed_single_batch(self, texts: List[str]):
        """Embed a list of texts using the in-process model/tokenizer (shared)."""
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded)
        emb = _mean_pooling_from_output(model_output, encoded["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)  # shape (batch, dim)
        return emb.cpu().numpy()  # (batch, dim) numpy array

    def embed_chunks(self, chunks: List[str], batch_size: int = 8, use_processes: bool = False):
        """
        Main entrypoint to embed a list of chunks.
        - batch_size: how many texts to process per worker call
        - use_processes: if True, uses ProcessPoolExecutor and each worker loads model
                         if False, uses ThreadPoolExecutor and shares model/tokenizer
        Returns: list of numpy arrays (each vector)
        """
        if not chunks:
            logger.warning("embed_chunks called with empty chunks.")
            return []

        self.chunks = list(chunks)
        self.all_embeddings = []

        # create batches
        chunk_groups = [self.chunks[i : i + batch_size] for i in range(0, len(self.chunks), batch_size)]

        if use_processes:
            logger.info("Embedding using ProcessPoolExecutor (each worker loads its own model).")
            # Prepare args for workers: (model_name, chunk_group, device_str)
            device_str = str(self.device)
            worker_args = [(self.model_name, grp, device_str) for grp in chunk_groups]
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(embed_chunk_group_worker, arg) for arg in worker_args]
                for future in as_completed(futures):
                    try:
                        result = future.result()  # result: list of list-vectors
                        for v in result:
                            self.all_embeddings.append(np.asarray(v, dtype=float))
                    except Exception as e:
                        logger.error(f"Error embedding chunk group (process): {e}")
        else:
            # Threads: share loaded model/tokenizer in-process
            logger.info("Embedding using ThreadPoolExecutor (shared model/tokenizer).")
            # Ensure model/tokenizer loaded
            if self.model is None or self.tokenizer is None:
                self.load_model()

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._embed_single_batch, grp) for grp in chunk_groups]
                for future in as_completed(futures):
                    try:
                        batch_emb = future.result()  # numpy array (batch, dim)
                        for v in batch_emb:
                            self.all_embeddings.append(np.asarray(v, dtype=float))
                    except Exception as e:
                        logger.error(f"Error embedding chunk group (thread): {e}")

        if not self.all_embeddings:
            logger.warning("No embeddings were created.")
            return []

        # Store to Qdrant (or log)
        if self.qdrant_enabled:
            vector_size = int(self.all_embeddings[0].shape[0])
            self.qdrant.recreate_collection(self.collection_name, vector_size)
            payloads = [{"text": chunk} for chunk in self.chunks]
            vectors = [vec.tolist() for vec in self.all_embeddings]
            self.qdrant.add_vector_embeddings(self.collection_name, vectors, payloads)
            logger.info(f"Stored {len(vectors)} vectors in Qdrant.")
        else:
            logger.info(f"Stored {len(self.all_embeddings)} vectors in memory (Qdrant disabled).")

        return self.all_embeddings

    # ---------- Query helpers ----------
    def get_user_query_embedding(self, query: str):
        """Return numpy vector for query (1D numpy array)."""
        if self.model is None or self.tokenizer is None:
            self.load_model()
        encoded = self.tokenizer(query, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded)
        emb = _mean_pooling_from_output(model_output, encoded["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)
        return emb[0].cpu().numpy()

    def get_similar_chunks(self, query_embedding: np.ndarray, top_k: int = 3):
        """Search Qdrant if enabled, else use in-memory cosine similarity."""
        if self.qdrant_enabled:
            logger.info("Searching via Qdrant...")
            results = self.qdrant.search_vectors(self.collection_name, query_embedding.tolist(), top_k=top_k)
            top_k_texts = [r["payload"]["text"] for r in results]
            top_k_scores = [r["score"] for r in results]
            return top_k_scores, top_k_texts

        if not self.all_embeddings:
            raise ValueError("No embeddings found. Please embed chunks first.")
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity([query_embedding], self.all_embeddings)[0]
        idxs = np.argsort(sims)[-top_k:][::-1]
        top_k_scores = [float(sims[i]) for i in idxs]
        top_k_texts = [self.chunks[i] for i in idxs]
        return top_k_scores, top_k_texts
