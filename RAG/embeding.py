from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from RAG.vector import QdrantClientWrapper
from config.logger_config import logger

class Embedder:
    def __init__(self, collection_name: str = "collection_1"):
        self.model = None
        self.tokenizer = None
        self.all_embeddings = []
        self.chunks = []
        self.device = torch.device("cpu")
        self.collection_name = collection_name

        # Try to connect to Qdrant
        try:
            self.qdrant = QdrantClientWrapper(url="http://localhost:6333")
            self.qdrant_enabled = True
            logger.info(f"Qdrant connected. Using collection: {self.collection_name}")
        except Exception as e:
            self.qdrant = None
            self.qdrant_enabled = False
            logger.warning(f"Qdrant not available ({e}). Falling back to in-memory search.")

        logger.info(f"Using device: {self.device}")
        self.load_model()

    # ---------- MODEL ----------
    def load_model(self):
        model_name = "BAAI/bge-m3"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        logger.info(f"Loaded model {model_name} on device {self.device}")
        self.model.eval()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    # ---------- EMBEDDING ----------
    def embed_chunks(self, chunks):
        if self.model is None or self.tokenizer is None:
            self.load_model()

        self.all_embeddings = []
        self.chunks = chunks

        for chunk in chunks:
            encoded_input = (
                self.tokenizer(chunk, padding=True, truncation=True, return_tensors="pt")
                .to(self.device)
            )
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            embeddings = self.mean_pooling(model_output, encoded_input["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            self.all_embeddings.append(embeddings[0].cpu().numpy())

        # Store in Qdrant if available
        if self.qdrant_enabled and self.all_embeddings:
            vector_size = len(self.all_embeddings[0])
            self.qdrant.recreate_collection(self.collection_name, vector_size)
            payloads = [{"text": chunk} for chunk in self.chunks]
            self.qdrant.add_vector_embeddings(self.collection_name, self.all_embeddings, payloads)
            logger.info(f"Stored {len(self.all_embeddings)} vectors in Qdrant.")
        else:
            logger.info(f"Stored {len(self.all_embeddings)} vectors in memory (Qdrant disabled).")

        return self.all_embeddings

    def get_user_query_embedding(self, query):
        if self.model is None or self.tokenizer is None:
            self.load_model()

        encoded_input = (
            self.tokenizer(query, padding=True, truncation=True, return_tensors="pt").to(self.device)
        )
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = self.mean_pooling(model_output, encoded_input["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings[0].cpu().numpy()

    # ---------- RETRIEVAL ----------
    def get_similar_chunks(self, query_embedding, top_k=3):
        """Search Qdrant if enabled, else use in-memory cosine similarity."""
        if self.qdrant_enabled:
            logger.info("Searching via Qdrant...")
            results = self.qdrant.search_vectors(self.collection_name, query_embedding, top_k=top_k)
            top_k_texts = [r["payload"]["text"] for r in results]
            top_k_scores = [r["score"] for r in results]
            return top_k_scores, top_k_texts

        # Fallback to in-memory search
        if not self.all_embeddings:
            raise ValueError("No embeddings found. Please embed chunks first.")
        similarities = cosine_similarity([query_embedding], self.all_embeddings)[0]
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        top_k_scores = [float(similarities[i]) for i in top_k_indices]
        top_k_texts = [self.chunks[i] for i in top_k_indices]
        return top_k_scores, top_k_texts