# Template DDL for Corpus Table

CORPUS_DDL = """
                CREATE TABLE IF NOT EXISTS {schema}.{table_name} (
                        id serial4 NOT NULL,
                        pubmed_id varchar(20) NULL,
                        title text NULL,
                        "content" text NULL,
                        is_test bool DEFAULT true NULL,
                        is_available bool DEFAULT true NULL,
                        CONSTRAINT {table_name}_pkey PRIMARY KEY (id),
                        CONSTRAINT {table_name}_pubmed_id_key UNIQUE (pubmed_id)
                    );
                    """
# Template DDL for the Vector Store Tables
STORE_DDL = """
CREATE TABLE IF NOT EXISTS {schema}.{table_name} (
    id           SERIAL4 PRIMARY KEY,
    pubmed_id    VARCHAR(20) NOT NULL,
    chunk        INT4        NOT NULL,
    embedding    VECTOR({vector_dim}) NULL,
    content      TEXT        NOT NULL,
    beginoffset  INT4,
    endoffset    INT4,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_{table_name}_pubmed_id FOREIGN KEY (pubmed_id)
    REFERENCES {schema}.{corpus_table}(pubmed_id)
    ON DELETE CASCADE
);
"""

FTS_STORE_DDL = """
CREATE TABLE IF NOT EXISTS {schema}.{table_name} (
    id SERIAL PRIMARY KEY,
    pubmed_id VARCHAR(20) NOT NULL,
    chunk INT NOT NULL,
    content TEXT NOT NULL,
    cleansed_content text not null,
    fts_content TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', cleansed_content)
    ) STORED,
    beginoffset INT,
    endoffset INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_{table_name}_pubmed_id FOREIGN KEY (pubmed_id)
    REFERENCES {schema}.{corpus_table}(pubmed_id)
    ON DELETE CASCADE
);
"""
# Template DDLS to create and rebuild the table indexes

IVFFLAT_INDEX_DDL = """
DROP INDEX IF EXISTS {schema}.idx_{table_name}_embedding;
SET maintenance_work_mem = '1000MB';
CREATE INDEX idx_{table_name}_embedding
ON {schema}.{table_name}
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = {lists});
"""

HNSW_INDEX_DDL = """
DROP INDEX IF EXISTS {schema}.idx_{table_name}_embedding;
SET maintenance_work_mem = '1000MB';
CREATE INDEX idx_{table_name}_embedding
ON {schema}.{table_name}
USING hnsw (embedding vector_cosine_ops)
WITH (m = {m}, ef_construction = {ef_construction});
"""

FTS_INDEX_DDL = """
DROP INDEX IF EXISTS {schema}.idx_{table_name}_fts;
CREATE INDEX idx_{table_name}_fts
ON {schema}.{table_name}
USING GIN (fts_content);
"""