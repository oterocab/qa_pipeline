from abc import ABC, abstractmethod
from typing import List

class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed_query(self, query: str):
        """Method to embed a query."""
        pass

    @abstractmethod
    def embed_queries(self, queries: List[str]): # In most implementatios will be used as an alias for embed_documents
        """Method to embed a batch of queries."""
        pass

    @abstractmethod
    def embed_document(self, document: str):
        """Method to embed a document or text chunk."""
        pass

    @abstractmethod
    def embed_documents(self, document: List[str]):
        """Method to embed a batch of documents or text chunks."""
        pass