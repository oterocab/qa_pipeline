from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from .base_connection import BaseConnectionHandler
from typing import List, Tuple, Dict


class BasePostgreSQLConnectionHandler(BaseConnectionHandler):
    def __init__(self, config, logger=None):
        """
        Initializes the BasePostgreSQLConnectionHandler with a configuration dictionary.

        :param config: A dictionary containing database connection details.
        :param logger: A logger instance (optional).
        """
        super().__init__(config, logger)
        
        try:
            self.pool = ConnectionPool(
                conninfo=f"dbname={self.db_name} user={self.db_user} password={self.db_password} "
                         f"host={self.db_host} port={self.db_port}",
                min_size=self.min_pool_size,
                max_size=self.max_pool_size,
            )
            self.logger.info(f"PostgreSQL connection pool initialized with min_size={self.min_pool_size}, max_size={self.max_pool_size}.")
        except Exception as e:
            self.logger.error(f"Error initializing PostgreSQL connection pool: {e}")
            raise
    
    def close_pool(self):
        """
        Closes the PostgreSQL connection pool.
        """
        if self.pool:
            self.pool.close()
            self.logger.info("PostgreSQL connection pool closed.")

    
    def get_document(self, pubmed_id:str, table_name: str):
        try:
            with self.pool.connection() as connection:
                with connection.cursor(row_factory=dict_row) as cur:
                    query = f"""
                        SELECT * FROM {self.db_schema}.{table_name}
                        WHERE pubmed_id = %s
                    """
                    cur.execute(query, (pubmed_id,))
                    document = cur.fetchone()

                return document
            
        except Exception as e:
            self.logger.error(f"Error while retrieving document with id {pubmed_id}: {str(e)}")
            raise
    
            
    def store_document(self, doc_id:str, doc_title:str, doc_body:str, is_available: bool, is_test: bool = False)-> None:
        """
        Store the documents in the base table, which will be reused in all embedding generation processes.

        :param doc_id: Unique document ID.
        :param doc_title: Title of the document.
        :param doc_body: Body/content of the document.
        :param is_available: Could be fetched from origin (Content is not restricted).
        :param is_test: Document is only used in the evaluation files.
        """
        try:
            with self.pool.connection() as connection:
                with connection.cursor(row_factory=dict_row) as cur:
                    query=f"""
                        INSERT INTO {self.db_schema}.documents (pubmed_id, title, content, is_available, is_test)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (pubmed_id) DO NOTHING
                            """
                    cur.execute(query,(doc_id, doc_title, doc_body, is_available, is_test))
                    connection.commit()

                    if cur.rowcount == 0:
                        self.logger.info(f"Document with ID {doc_id} already exists. Skipped insertion.")
                    else:
                        self.logger.info(f"Inserted document with ID {doc_id} successfully.")

        except Exception as e:
            self.logger.error(f"Error while inserting document {doc_id}: {e}")
            raise e
        
    def get_all_document_ids(self, table_name: str) -> List[str]:
        """
        Get all documents ids.

        :param table_name: target table to fetch the ids.
        """
        try:
            with self.pool.connection() as connection:
                with connection.cursor(row_factory=dict_row) as cur:
                    query = f"""
                        SELECT pubmed_id FROM {self.db_schema}.{table_name}
                        WHERE is_available = TRUE
                    """
                    cur.execute(query,)
                    results = cur.fetchall()
                    return results
        except Exception as e:
            self.logger.error(f"Error getting list of documents present in database: {e}")
            raise e
        
    def get_documents_by_ids(self, document_ids: List[int], table_name: str) -> List[Dict]:
        """
        Given a list of ids, fetchs the data from the database

        :param document_ids: document ids of the targets docs.
        :param table_name: target table to fetch the ids.
        """
        try:
            with self.pool.connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    if isinstance(document_ids[0], dict):
                        document_ids = [doc["pubmed_id"] for doc in document_ids]
                    query = f"""
                        SELECT pubmed_id, title, content
                        FROM {self.db_schema}.{table_name}
                        WHERE pubmed_id = ANY(%s)
                    """
                    cur.execute(query, (document_ids,))
                    rows = cur.fetchall()

            return rows

        except Exception as e:
            self.logger.error(f"Failed to fetch batch documents: {str(e)}")
            raise e

        
    def get_existing_pmids(self, pmid_list, table_name):
        """
        Given a list of ids, fetchs the ones already present in the DB

        :param document_ids: document ids of the docs.
        :param table_name: target table to fetch the ids.
        """
        try:
            with self.pool.connection() as connection:
                with connection.cursor(row_factory=dict_row) as cur:
                    query = f"""
                        SELECT pubmed_id FROM {self.db_schema}.{table_name}
                        WHERE pubmed_id = ANY(%s)
                    """
                    cur.execute(query, (pmid_list,))
                    results = cur.fetchall()
                    return results
        except Exception as e:
            self.logger.error(f"Error getting list of documents present in database: {e}")
            raise e
        

    def insert_embedding(self, document_id: str, chunk_number: int, embedding: list[float], content, beginoffset: int, endofsset: int, table_name:str):
        """
        inserts the document embedding, content and related metadata i nthe target table

        :param document_ids: document ids of the targets docs.
        :param table_name: target table to fetch the ids."""
        try:
            with self.pool.connection() as connection:
                with connection.cursor() as cur:
                    query = f"""
                        INSERT INTO {self.db_schema}.{table_name} (pubmed_id, chunk, embedding, content, beginoffset, endoffset)
                        VALUES (%s, %s, %s::vector, %s, %s, %s)
                    """
                    cur.execute(query, (document_id, chunk_number, embedding, content, beginoffset, endofsset))
                    connection.commit()
                    self.logger.info(f"Embedding for document_id '{document_id}' inserted successfully.")
        except Exception as e:
            self.logger.error(f"Error while inserting embedding: {str(e)}")
            raise


    def insert_embeddings(self, records: List[Tuple], table_name: str):
        """
        inserts the document embedding, content and related metadata into the target table
        for a batch of documents.

        :param records: document records with embeddings and metadata to be inserted
        :param table_name: target table to insert the records.
        """
        try:
            with self.pool.connection() as connection:
                with connection.cursor() as cur:
                    values_str = ",".join(["(%s, %s, %s, %s, %s::vector, %s)"] * len(records))
                    flat_values = [item for record in records for item in record]
                    query = f"""
                        INSERT INTO {self.db_schema}.{table_name} (pubmed_id, chunk, beginoffset, endoffset, embedding, content)
                        VALUES {values_str}
                    """
                    cur.execute(query, flat_values)
                    connection.commit()
                    self.logger.info(f"Inserted {len(records)} embeddings.")
        except Exception as e:
            self.logger.error(f"Embedding Batch insert failed: {str(e)}")
            raise

    def insert_fts_vectors(self, records: List[Tuple], table_name: str):
        """
        inserts the document into the target table and creates the fts representation, content and related metadata into the target table
        for a batch of documents.

        :param records: document records with embeddings and metadata to be inserted
        :param table_name: target table to insert the records.
        """
        try:
            with self.pool.connection() as connection:
                with connection.cursor() as cur:
                    values_str = ",".join(["(%s, %s, %s, %s, %s)"] * len(records))
                    flat_values = [item for record in records for item in record]
                    query = f"""
                        INSERT INTO {self.db_schema}.{table_name} (pubmed_id, chunk, beginoffset, endoffset, content)
                        VALUES {values_str}
                    """
                    cur.execute(query, flat_values)
                    connection.commit()
                    self.logger.info(f"Inserted {len(records)} embeddings.")
        except Exception as e:
            self.logger.error(f"FTS Batch insert failed: {str(e)}")
            raise
    
    def get_top_k_docs(self, query_embedding: list, table_name: str, top_k: int) -> List[Dict]:
        """
        Retrieve the top-k most relevant documents from the specified table based on cosine similarity 
        between the provided query embedding and the stored document embeddings

        :param query_embedding: pre-calculated query embedding
        :param table_name: target store table with the embeddings.
        """
        try:
            with self.pool.connection() as connection:
                with connection.cursor(row_factory=dict_row) as cur:
                    embedding_query = f"""
                    SELECT emb.pubmed_id AS id, 
                           emb.content AS content, 
                           emb.chunk as chunk, 
                           1 - cosine_distance(emb.embedding, %s::vector) as score,
                           emb.beginoffset as beginoffset,
                           emb.endoffset as endoffset
                    FROM {self.db_schema}.{table_name} emb
                    ORDER BY score DESC
                    LIMIT %s;
                    """

                    cur.execute(embedding_query, (query_embedding, top_k))
                    embedding_results = cur.fetchall()

                    return embedding_results
        except Exception as e:
            self.logger.error(f"Error fetching embeddings from database: {str(e)}")
            raise


    def get_top_k_docs_fts(self, query: str, table_name: str, top_k: int) -> List[Dict]:
        """
        Retrieve the top-k documents from the specified table using full text search
        capabilites from PostgreSQL db. Using function phraseto_tsquery() to search for
        lexeme sequences.

        :param query_embedding: pre-calculated query embedding
        :param table_name: target store table with the fts vector representations.
        """
        try:
            with self.pool.connection() as connection:
                with connection.cursor(row_factory=dict_row) as cur:
                    fts_query = f"""SELECT pubmed_id AS id,
                                           chunk,
                                           content,
                                           ts_rank(fts_content, phraseto_tsquery ('english', %s)) AS score,
                                           beginoffset,
                                           endoffset
                                    FROM {self.db_schema}.{table_name}
                                    WHERE fts_content @@ phraseto_tsquery ('english', %s)
                                    ORDER BY score DESC
                                    LIMIT %s;
                                    """
                    cur.execute(fts_query, (query, query, top_k))
                    fts_results = cur.fetchall()
                    return fts_results
        except Exception as e:
            self.logger.error(f"Error fetching fts results from database: {str(e)}")
            raise
