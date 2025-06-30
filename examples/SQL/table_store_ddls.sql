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

-- Set up and IVF index on the embedding column to speed up retrieval! Set up N value looking for a trade off between speed and recall

CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = n);

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

-- Evaluation procedure for IVFF and HNSW execution times

DO $$
DECLARE
    t0  timestamptz;
    t1  timestamptz;
    tot_ivf_ms  double precision := 0;
    tot_hnsw_ms double precision := 0;
    loops       int := 1000;
BEGIN
    FOR i IN 1..loops LOOP

        t0 := clock_timestamp();

        SET LOCAL ivfflat.probes = 60;      -- Tune search value

        PERFORM embedding, content
        FROM   schema.table_with_ivff_index
        ORDER  BY embedding <=> '[...]'::vector
        LIMIT  100;

        t1 := clock_timestamp();
        tot_ivf_ms := tot_ivf_ms + EXTRACT(epoch FROM (t1 - t0)) * 1000;


        t0 := clock_timestamp();

        SET LOCAL hnsw.ef_search = 40;     -- Tune search value

        PERFORM embedding, content
        FROM   schema.table_with_hnsw_index
        ORDER  BY embedding <=> '[...]'::vector
        LIMIT  100;

        t1 := clock_timestamp();
        tot_hnsw_ms := tot_hnsw_ms + EXTRACT(epoch FROM (t1 - t0)) * 1000;
    END LOOP;

    RAISE NOTICE 'Iters: %', loops;
    RAISE NOTICE 'Mean IVFFLAT  (ms): %.3f',  tot_ivf_ms  / loops;
    RAISE NOTICE 'Mean HNSW     (ms): %.3f',  tot_hnsw_ms / loops;
END $$;
select count(*) from tfm_schema.ft_bge_small_en_v15_ivff fbsevh ;