from FlagEmbedding import FlagAutoReranker
from typing import List, Tuple
from .base_reranker_model import BaseReranker

class FlagEmbeddingReranker(BaseReranker):
    def __init__(self, **kwargs):
        """
        Initialize the reranker model using fine-tuned checkpoint.
        Expects model_name_or_path and other tokenizer/model args.
        """
        self.model = FlagAutoReranker.from_finetuned(**kwargs)

    def compute_scores(self, query: str, documents: List[str], normalize: bool = False) -> List[float]:
        """
        Compute relevance scores between the query and each document.
        Returns a list of floats.
        """
        if not documents:
            return []

        pairs = [[query, doc] for doc in documents]

        assert all(isinstance(p, list) and len(p) == 2 for p in pairs), "Each input must be a [query, doc] pair."

        return self.model.compute_score(pairs, normalize=normalize)

    def rerank(self, query: str, documents: List[Tuple], top_k: int, **kwargs):
        """
        Reranks the documents using the model set in the class attribute property, setting a new score for each document.
        """
        documents = documents
        texts = [doc["content"] for doc in documents]
        scores = self.compute_scores(query, texts, **kwargs)

        reranked = []
        for doc, rerank_score in zip(documents, scores):
            reranked.append({
                "id": doc["id"],
                "content": doc["content"],
                "chunk": doc["chunk"],
                "score": doc["score"],
                "rerank_score": rerank_score,
                "beginoffset": doc["beginoffset"],
                "endoffset": doc["endoffset"]
            })

        return sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)[:top_k]


