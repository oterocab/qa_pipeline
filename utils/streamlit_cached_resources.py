import streamlit as st
from db_connectors.postgres_connection import BasePostgreSQLConnectionHandler
from retrievers.base_retriever import BaseDocumentRetriever
from readers.reader_factory import ChatModelFactory
from evaluation.reader_evaluation import RagasEvaluator
from readers.base_reader import BaseReader



# Functions used to cache resources and data in Streamlit

@st.cache_resource(show_spinner=False)
def get_reader(system_prompt: str, provider_name: str, model_config_kargs: dict):
    return ChatModelFactory.get_model(
        system_prompt,
        provider_name,
        model_config_kargs=model_config_kargs
    )

@st.cache_resource(show_spinner=False)
def get_db_conn(conn_config: dict):
    return BasePostgreSQLConnectionHandler(conn_config)

@st.cache_resource(show_spinner=False)
def get_retriever(_db_conn, corpus_table: str, store_table: str, embedder_cfg: dict, reranker_cfg: dict, index_config: dict, _hash_marker: str):
    return BaseDocumentRetriever(
        _db_conn,
        corpus_table,
        store_table,
        embedding_config=embedder_cfg,
        reranker_config=reranker_cfg,
        store_index_config=index_config
    )
@st.cache_resource(show_spinner=False)
def get_ragas_evaluator(_retriever: BaseDocumentRetriever, _reader: BaseReader, search_top_k: int, 
                        llm_top_k: int, reranker_top_k: int, use_reranker: bool, fts_search: bool) -> RagasEvaluator:
    return RagasEvaluator(
        retriever=_retriever,
        reader=_reader,
        search_top_k=search_top_k,
        llm_top_k=llm_top_k,
        reranker_top_k=reranker_top_k,
        use_reranker=use_reranker,
        fts_search=fts_search
    )
@st.cache_data(show_spinner=False)
def cached_retrieve_docs( _retriever: BaseDocumentRetriever, query: str, top_k: int, rerank_top_k: int, 
                            use_reranking: bool, embedder_config: dict, reranker_config: dict):
    """Cache based on configs, not retriever instance itself."""
    rerank_latency = 0
    _ = (str(embedder_config), str(reranker_config))
    if not embedder_config:
        docs, db_latency = _retriever.get_top_k_docs_fts(query, top_k)
    else:
        docs, db_latency = _retriever.get_top_k_docs(query, top_k)

    if use_reranking:
        docs, rerank_latency = _retriever.rerank_top_k_docs(query, docs, rerank_top_k, normalize=True)
        
    return docs, db_latency, rerank_latency