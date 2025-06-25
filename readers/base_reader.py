from abc import ABC, abstractmethod
from typing import List

class BaseReader(ABC):
    @abstractmethod
    def answer(self, query: str, documents: List[str]) -> str:
        """Generate an answer from the query and the context created from the retrieved documents."""
        pass
    
    @abstractmethod
    def build_context(self, documents: List[str]) -> str:
        "Generate context to pass to the document reader"
        pass