from .reranker_models import FlagEmbeddingReranker

class RerankerFactory:
    @staticmethod
    def get_reranker_model( reranking_config: dict):
        """
        Returns the reranker model instance
        """
        
        return FlagEmbeddingReranker(**reranking_config) 