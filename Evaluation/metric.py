import evaluate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pandas as pd

class TextMetrics:
    def __init__(self):
        self.bleu_metric = evaluate.load("bleu")
        self.rouge_metric = evaluate.load("rouge")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.vectorizer = TfidfVectorizer()

    def compute_metrics(self, text1: str, text2: str):
        # ---- BLEU ----
        bleu_res = self.bleu_metric.compute(
            predictions=[text2],
            references=[text1]
        )
        # ---- ROUGE ----
        rouge_res = self.rouge_metric.compute(
            predictions=[text2],
            references=[text1]
        )
        # ---- Embeddings ----
        emb = self.model.encode([text1, text2])
        cosine_sim = float(cosine_similarity([emb[0]], [emb[1]])[0][0])
        euclidean_dist = float(euclidean_distances([emb[0]], [emb[1]])[0][0])
        # ---- TF-IDF Cosine ----
        try:
            tfidf_vecs = self.vectorizer.fit_transform([text1, text2])
            tfidf_cosine = float(cosine_similarity(tfidf_vecs[0], tfidf_vecs[1])[0][0])
        except:
            tfidf_cosine = 0.0

        return {
            "text1": text1,
            "text2": text2,
            "BLEU": bleu_res["bleu"],
            "ROUGE-1": rouge_res["rouge1"],
            "ROUGE-L": rouge_res["rougeL"],
            "CosineSim": cosine_sim,
            "EuclideanDist": euclidean_dist,
            "TFIDF_Cosine": tfidf_cosine
        }

    def batch_compute(self, pairs, return_df=False):
        """
        pairs: list of [text1, text2]
        returns dict list OR pandas dataframe
        """
        results = []
        for t1, t2 in tqdm(pairs, desc="Computing metrics"):
            res = self.compute_metrics(t1, t2)
            results.append(res)
        if return_df:
            return pd.DataFrame(results)
        return results