import os
import yaml
from dotenv import load_dotenv
from typing import Dict, Any, Optional
import logging

class BaseAppConfig:
    """Handles application configuration."""

    def __init__(self, embedder_config_file: str = None, reranker_config_file: str = None, reader_config_file:str = None, tokenizer_config_file: str = None, logger = None):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self._env_file_path:str = os.path.join(project_root,".env")
        if os.path.exists(self._env_file_path):
            load_dotenv(dotenv_path=self._env_file_path)
        else:
            raise FileNotFoundError(f".env file not found at {self._env_file_path}")
        
        self._conn_config:dict[str, Any] = None
        self._embedder_config_file: str = embedder_config_file or "retrieval_config/embedders.yaml"
        self._embedder_config: Optional[Dict[str, Any]] = None
        self._reranker_config_file : str = reranker_config_file or "retrieval_config/rerankers.yaml"
        self._reranker_config: Optional[Dict[str, Any]] = None
        self._reader_config_file: str = reader_config_file or "retrieval_config/readers.yaml"
        self._reader_config: Optional[Dict[str, Any]] = None
        self._tokenizer_config_file: str = tokenizer_config_file or "retrieval_config/tokenizers.yaml"
        self._tokenizer_config: Optional[Dict[str, Any]] = None
        self._ncbi_api_key:str = None
        self._evaluator_llm_name: str = None
        self.logger = logger if logger else logging.getLogger(__name__)
    
    @property
    def conn_config(self) -> Dict[str, Any]:
        if self._conn_config is None:
            self._conn_config = {
                'db_name': os.getenv('DB_NAME'),
                'db_schema': os.getenv('DB_SCHEMA'),
                'db_user': os.getenv('DB_USER'),
                'db_password': os.getenv('DB_PASSWORD'),
                'db_host': os.getenv('DB_HOST'),
                'db_port': os.getenv('DB_PORT'),
                'min_pool_size': int(os.getenv('MIN_SQL_POOL_SIZE_BOT', 10)),
                'max_pool_size': int(os.getenv('MAX_SQL_POOL_SIZE_BOT', 20)),
            }
            for key, value in self._conn_config.items():
                if not value:
                    raise ValueError(f"Environment variable '{key}' is not set.")
        return self._conn_config
    
    @property
    def embedder_config(self) -> Dict[str, Any]:
        if self._embedder_config is None:
            config_dir = self._get_env("CONFIG_DIR")
            if not self._embedder_config_file:
                raise ValueError("Embedder config file name is not set.")

            full_path = os.path.join(config_dir, self._embedder_config_file)
            yaml_data = self._load_yaml(full_path)

            self._embedder_config = yaml_data["embedders"]
        return self._embedder_config

    @property
    def reranker_config(self) -> Dict[str, Any]:
        if self._reranker_config is None:
            config_dir = self._get_env("CONFIG_DIR")
            if not self._reranker_config_file:
                raise ValueError("Retriever config file name is not set.")

            full_path = os.path.join(config_dir, self._reranker_config_file)
            yaml_data = self._load_yaml(full_path)

            self._reranker_config = yaml_data["rerankers"]
        return self._reranker_config
    
    @property
    def reader_config(self) -> Dict[str, Any]:
        if self._reader_config is None:
            config_dir = self._get_env("CONFIG_DIR")
            if not self._reader_config_file:
                raise ValueError("Reader config file name is not set.")

            full_path = os.path.join(config_dir, self._reader_config_file)
            yaml_data = self._load_yaml(full_path)

            self._reader_config = yaml_data["readers"]
        return self._reader_config

    @property
    def tokenizer_config(self) -> Dict[str, Any]:
        if self._tokenizer_config is None:
            config_dir = self._get_env("CONFIG_DIR")
            if not self._tokenizer_config_file:
                raise ValueError("Tokenizer config file name is not set.")

            full_path = os.path.join(config_dir, self._tokenizer_config_file)
            yaml_data = self._load_yaml(full_path)

            self._tokenizer_config = yaml_data["tokenizers"]
        return self._tokenizer_config
    
    @property
    def ncbi_api_key(self) -> str:
        if self._ncbi_api_key is None:
            self._ncbi_api_key = os.getenv('NCBI_API_KEY')
        return self._ncbi_api_key
    
    @property
    def evaluator_llm_name(self) -> str:
        if self._evaluator_llm_name is None:
            self._evaluator_llm_name = os.getenv('EVALUATOR_LLM_NAME')
        return self._evaluator_llm_name


    def _get_env(self, key: str, default: Optional[str] = None) -> str:
        value = os.getenv(key, default)
        if value is None:
            raise ValueError(f"Environment variable '{key}' is not set and has no default.")
        return value

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"YAML configuration file not found: {path}")
        with open(path, "r") as f:
            return yaml.safe_load(f)