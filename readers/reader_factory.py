from .openai_chat_reader import OpenAIReader
from .anthropic_chat_reader import AnthropicReader

class ChatModelFactory:
    @staticmethod
    def get_model(system_prompt: str, provider_name: str, model_config_kargs:dict):
        """
        Factory to instantiate the reader class object based on the provided configuration
        """
        if provider_name.lower() == "openai":
            return OpenAIReader(system_prompt, model_config_kargs)
        elif provider_name.lower() == "anthropic":
            return AnthropicReader(system_prompt, model_config_kargs)
        else:
            raise ValueError(f"Model {provider_name} is not supported.")