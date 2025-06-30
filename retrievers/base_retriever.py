from .embedding.embedding_factory import EmbeddingFactory
from .reranking.reranker_factory import RerankerFactory
from .text_splitters.text_splitter_factory import TextSplitterFactory
from db_connectors.postgres_connection import BasePostgreSQLConnectionHandler

import logging
from typing import Dict, List, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize
import unicodedata
import re
import time


class BaseDocumentRetriever:
    """Responsible for processing, embedding, and inserting documents into the target store, in this case the Postgresql DB."""

    def __init__(self, store_handler: BasePostgreSQLConnectionHandler, corpus_table:str, store_table:str, 
                 text_splitter_config=None, embedding_config: Dict = None, reranker_config: Dict=None, 
                 store_index_config: Dict=None, embedding_dim: int = None, logger=None):
        self.store_handler = store_handler
        self.embedding_model = EmbeddingFactory.get_embedding_model(embedding_config) if embedding_config else None
        self.reranker_model = RerankerFactory.get_reranker_model(reranker_config) if reranker_config else None
        self.corpus_table = corpus_table
        self.store_table = store_table
        self.embedding_dim = embedding_dim
        self.store_index_config = store_index_config
        self.text_splitter = TextSplitterFactory.get_text_splitter(text_splitter_config)
        self.logger = logger if logger else logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    
    def index_documents(self, mode: str = "embedding", batch_size: int = 500, index_rebuild: bool=False, truncate_store: bool=False):
        """Index the documents into the store table, using the provided configuration in the class instance and 
        the function parameters. The mode defines if its gonna be indexed for dense retrieval using embeddings (mode= embedding)
        or rather will be indexed for PostgreSQL full text search."""
        try:
            if mode not in {"embedding", "fts"}:
                raise ValueError("Invalid mode. Use 'embedding' or 'fts'.")

            
            self.store_handler.initialize_store(self.corpus_table, self.store_table, self.embedding_dim, mode)
            if truncate_store:
                self.store_handler.truncate_store(self.store_table)
            doc_ids = list(self.store_handler.get_not_indexed_ids(self.corpus_table, self.store_table))

            with tqdm(total=len(doc_ids), desc=f"Processing documents ({mode})", dynamic_ncols=True) as pbar:
                for i in range(0, len(doc_ids), batch_size):
                    batch = doc_ids[i:i + batch_size]
                    chunks = self.prepare_document_chunks(batch)

                    if not chunks:
                        continue

                    if mode == "embedding":
                        texts = [chunk["content"] for chunk in chunks]
                        embeddings = self.embedding_model.embed_documents(texts)

                        embed_data = [
                            (
                                chunk["pubmed_id"],
                                chunk["chunk"],
                                chunk["beginoffset"],
                                chunk["endoffset"],
                                embedding,
                                chunk["content"]
                            )
                            for chunk, embedding in zip(chunks, embeddings)
                        ]

                        self.store_handler.insert_embeddings(embed_data, self.store_table)

                    elif mode == "fts":
                        chunks = [(chunk["pubmed_id"], 
                                   chunk["chunk"], 
                                   chunk["beginoffset"], 
                                   chunk["endoffset"], 
                                   chunk["content"],
                                   normalize_for_fts(chunk["content"])
                                   ) 
                                  for chunk in chunks]
                        self.store_handler.insert_fts_vectors(chunks, table_name=self.store_table)

                    pbar.update(len(batch))

            if doc_ids or index_rebuild:
                self.logger.info("Starting index build")
                index_start = time.time()
                self.store_handler.build_index(self.store_table, self.store_index_config, mode)
                index_end = time.time()- index_start
                self.logger.info(f"Index build completed after: {index_end:.2f}")

        except Exception as e:
            self.logger.error(f"Document processing failed in for store table {self.store_table}: {str(e)}")
            raise
        
    def get_top_k_docs(self, query: str, search_top_k: int):
        """Get top K documents using dense retrieval with cosine similarity search"""

        query_embedding = self.embedding_model.embed_query(query)
        top_k_docs, latency = self.store_handler.get_top_k_docs(query_embedding, self.store_table, search_top_k, self.store_index_config)

        return top_k_docs, latency
    
    def get_top_k_docs_fts(self, query: str, search_top_k: int):
        """Get top K documents using PostgreSQL full text search capabilities.
           The query needs to be preprocessed to enhace the search results"""
        
        terms_query = build_tsquery(normalize_for_fts(query))
        top_k_docs, latency = self.store_handler.get_top_k_docs_fts(terms_query, self.store_table, search_top_k)

        return top_k_docs, latency
    
    def get_top_k_docs_batch(self, queries: List[str], top_k: int, fts_search: bool = False, max_workers: int = 1) -> List[List[dict]]:
        """
        Retrieves top-K documents for a batch of queries concurrently.
        """

        if not fts_search:
            query_embeddings = self.embedding_model.embed_queries(queries)
        else:
            query_embeddings = [None] * len(queries)

        def get_similar_records(pos: int) -> Tuple[int, List[dict]]:
            query = queries[pos]
            if fts_search:
                docs, latency = self.get_top_k_docs_fts(build_tsquery(normalize_for_fts(query)), top_k)
            else:
                embedding = query_embeddings[pos]
                docs, latency = self.store_handler.get_top_k_docs(embedding, self.store_table, top_k, self.store_index_config)
            return pos, docs, latency

        results = [None] * len(queries)
        latencies = [0.0] * len(queries)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(get_similar_records, i): i for i in range(len(queries))}
            for future in as_completed(futures):
                pos, docs, latency = future.result()
                results[pos] = docs
                latencies[pos] = latency
        return results, latencies
    
    def rerank_top_k_docs(self, query: str, top_k_docs: List[Dict], rerank_top_k: int, **kwargs):
        top_k_docs, latency = self.reranker_model.rerank(query, top_k_docs, top_k=rerank_top_k, **kwargs)

        return top_k_docs, latency
    
    def prepare_document_chunks(self, doc_ids: List[str]) -> List[dict]:
        """Fetch documents by ID and return a list of chunk metadata dicts."""
        documents = self.store_handler.get_documents_by_ids(doc_ids, self.corpus_table)
        chunks = []

        for doc in documents:
            doc_id = doc['pubmed_id']
            title = (doc.get('title') or '').strip()
            content = (doc.get('content') or '').strip()

            if not content:
                self.logger.warning(f"Skipping document ID: {doc_id} (empty content)")
                continue

            doc_chunks = self.split_document(title, content)
            
            for chunk in doc_chunks:
                chunks.append({
                    "pubmed_id": doc_id,
                    "chunk": chunk['chunk_id'],
                    "content": chunk['text'],
                    "beginoffset": chunk['beginoffset'],
                    "endoffset": chunk['endoffset'],
                })

        return chunks

    
    def split_document(self, title: str, content: str) -> List[Dict]:
        chunks = []

        title = (title or "").strip()
        if title:
            chunks.append({
                "chunk_id": 0,
                "text": title,
                "beginoffset": 0,
                "endoffset": len(title)
            })

        content = (content or "").strip()
        if not content:
            return chunks

        content_chunks = self.text_splitter.split_text(content)
        current_offset = 0
        for pos, chunk_text in enumerate(content_chunks, start=1):
            begin = content.find(chunk_text, current_offset)

            if begin == -1:
                begin = content.find(chunk_text, max(0, current_offset - self.text_splitter._chunk_overlap))
                
            end = begin + len(chunk_text)

            chunks.append({
                "chunk_id": pos,
                "text": chunk_text,
                "beginoffset": begin,
                "endoffset": end
            })
            current_offset = end

        return chunks

def normalize_for_fts(query):
    "cleanses the text for full text search in PostgreSQL"
    query = query.lower()
    query = unicodedata.normalize("NFKD", query).encode("ascii", "ignore").decode("utf-8")
    query = re.sub(r"[^0-9a-z\s]", " ", query)
    query = re.sub(r"\s+", " ", query).strip()

    return query

def build_tsquery(query: str, operator: str = "|") -> str:
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(query.lower())
    tagged = pos_tag(tokens)

    words = [
        word for word, tag in tagged
        if tag.startswith("NN") and word.isalnum() and word not in stop_words
    ]
    if not words:
        words = [
        word for word in tokens
        if word.isalnum() and word not in stop_words]


    return f" {operator} ".join(words)