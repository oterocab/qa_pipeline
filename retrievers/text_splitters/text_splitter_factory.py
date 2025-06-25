from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from transformers import AutoTokenizer

class TextSplitterFactory:
    @staticmethod
    def get_text_splitter(splitter_config: dict):
        """
        Create a text splitter based on config.
        """
        if splitter_config:
            chunk_size = splitter_config.get("chunk_size", 400)
            chunk_overlap = splitter_config.get("chunk_overlap", 50)
            separators = splitter_config.get("separators", ["\n\n", "\n", ". ", " ", ""])
            tokenizer_name = splitter_config.get("tokenizer_name", None)

            if tokenizer_name:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
                return TokenTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)
