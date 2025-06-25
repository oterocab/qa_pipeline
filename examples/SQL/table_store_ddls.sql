-- Base table to store the Pubmed documents fecthed using the NCBI API.
CREATE TABLE schema.documents (
	id serial4 NOT NULL,
	pubmed_id varchar(20) NULL,
	title text NULL,
	"content" text NULL,
	is_test bool DEFAULT true NULL,
	is_available bool DEFAULT true NULL,
	CONSTRAINT documents_pkey PRIMARY KEY (id),
	CONSTRAINT documents_pubmed_id_key UNIQUE (pubmed_id)
);

-- Sample DDL for vector store table

CREATE TABLE schema.sample_embeddings (
	id serial4 NOT NULL,
	pubmed_id varchar(20) NOT NULL,
	chunk int4 NOT NULL,
	embedding vector(OUTPUT_DIM_OF_EMBEDDING_MODEL) NULL, -- Replace with the output dimension of the embedding model
	"content" text NOT NULL,
	beginoffset int4 NULL,
	endoffset int4 NULL,
	created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
	CONSTRAINT bge_small_en_embeddings_pkey PRIMARY KEY (id)
);

-- Is it possible and recommended to set up a index to speed up the similarity search

CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Sample DDL for PostgreSQL Full Text Search index table

CREATE TABLE tfm_schema.postgres_fts (
    id serial4 PRIMARY KEY,
    pubmed_id varchar(20) NOT NULL,
    chunk int4 NOT NULL,
    content text NOT NULL,
    fts_content tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    beginoffset int4 NULL,
    endoffset int4 NULL,
    created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL
);

--Can create a index to speed the retrival

CREATE INDEX idx_fts_content ON schema.postgres_fts USING GIN(fts_content);