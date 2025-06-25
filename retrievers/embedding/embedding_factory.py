from .embedding_models import OpenAIEmbeddingModel, FlagLocalEmbeddingModel
from .base_embedding_model import BaseEmbeddingModel
class EmbeddingFactory:
    @staticmethod
    def get_embedding_model(embedding_config: dict) -> BaseEmbeddingModel:
        """
        Returns an embedding model instance based on the configuration provided
        """
        config = embedding_config.copy()
        provider_name = config.pop("provider_name", "").lower()

        if not provider_name:
            return FlagLocalEmbeddingModel(**config)
        else:
            if provider_name.lower() == "openai":
                return OpenAIEmbeddingModel(**config)
            else:
                raise ValueError(f"Unsupported embedding provider: '{provider_name}'")
            


        
