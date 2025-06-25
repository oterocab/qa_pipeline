from FlagEmbedding import FlagAutoModel
from .base_embedding_model import BaseEmbeddingModel
from langchain_community.embeddings import OpenAIEmbeddings
from typing import List, Union
import numpy as np


class FlagLocalEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, **kwargs):
        """
        Initialize a local FlagEmbedding model from a fine-tuned checkpoint
        or use a model from their repository from the provided configuration.
        """
        self.model = FlagAutoModel.from_finetuned(**kwargs)

    def _ensure_list(self, vector: Union[np.ndarray, List[float]]) -> List[float]:
        """Requried to set an homogeneic output across all the classes implementing the BaseEmbeddingModel inferface"""
        return vector.tolist() if isinstance(vector, np.ndarray) else vector

    def embed_query(self, query: str) -> List[float]:
        vector = self.model.encode(query)
        return self._ensure_list(vector)
    
    def embed_queries(self, queries: List[str]) -> List[List[float]]:
        vectors = self.model.encode(queries)
        return [self._ensure_list(vec) for vec in vectors]

    def embed_document(self, document: str) -> List[float]:
        vector = self.model.encode(document)
        return self._ensure_list(vector)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        vectors = self.model.encode(documents)
        return [self._ensure_list(vec) for vec in vectors]


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """OpenAI Embedding Model implementation using LangChain's OpenAIEmbeddings."""

    def __init__(self, **kwargs):
        self.embeddings_model = OpenAIEmbeddings(**kwargs)

    def embed_query(self, query: str) -> List[float]:
        return self.embeddings_model.embed_query(query)
    
    def embed_queries(self, queries: List[str]) -> List[List[float]]:
        return self.embeddings_model.embed_documents(queries)

    def embed_document(self, document: str) -> List[float]:
        return self.embeddings_model.embed_documents([document])[0]

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return self.embeddings_model.embed_documents(documents)
