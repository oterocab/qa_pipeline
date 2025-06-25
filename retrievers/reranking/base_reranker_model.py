from abc import ABC, abstractmethod
from typing import List, Dict, Union

class BaseReranker(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    def compute_scores(self, query: str, documents: List[str], normalize: bool = False) -> List[float]:
        """
        Compute raw relevance scores between the query and each document.
        Args:
            query: The input query string.
            documents: List of document strings.
            normalize: Whether to apply sigmoid normalization to the scores.
        Returns:
            List of floats representing scores.
        """
        pass

    @abstractmethod
    def rerank(self, query: str, documents: List[str], normalize: bool = False) -> List[Dict[str, Union[str, float]]]:
        """
        Given a query and list of document texts, return a list of dicts like:
        [{"text": ..., "score": ...}, ...], sorted by score descending.
        Args:
            query: The input query string.
            documents: List of document strings.
            normalize: Whether to apply sigmoid normalization to the scores.
        Returns:
            Ranked list of documents with scores.
        """
        pass
