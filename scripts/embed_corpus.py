from retrievers.base_retriever import BaseDocumentRetriever
from db_connectors.postgres_connection import BasePostgreSQLConnectionHandler
from retrievers.text_splitters import TextSplitterFactory
from utils.app_config import BaseAppConfig
import argparse
import sys
from rich import print
from rich.prompt import Prompt

def main(args) -> None:
    config = BaseAppConfig(
        embedder_config_file=args.config
    )

    # Prompt for the embedder configuration to be used

    embedder_categories = list(config.embedder_config.keys())
    print(f"\n Available embedder categories: {embedder_categories}")
    selected_category = Prompt.ask("Choose a category", choices=embedder_categories)

    category_embedders = config.embedder_config[selected_category]
    embedder_names = [e["name"] for e in category_embedders]
    print(f"\n Available embedders in [{selected_category}]: {embedder_names}")
    selected_name = Prompt.ask("Choose an embedder", choices=embedder_names)

    selected_embedder = next(e for e in category_embedders if e["name"] == selected_name)
    store_table = selected_embedder["store_table"]
    text_splitter_cfg = selected_embedder.get("text_splitter_config", {})

    mode = "embedding" if selected_name != "fts" else "fts"

    db_conn = BasePostgreSQLConnectionHandler(config.conn_config)

    retriever = BaseDocumentRetriever(
        store_handler=db_conn,
        embedding_config=selected_embedder.get("model_config") if mode != "fts" else None,
        corpus_table=args.corpus_table,
        store_table=store_table,
        text_splitter_config=text_splitter_cfg,
        embedding_dim=selected_embedder.get("embedding_dim"),
        store_index_config=selected_embedder.get("index_config",{}))

    retriever.index_documents(mode=mode, batch_size=args.batch_size, index_rebuild=args.index_rebuild, truncate_store=args.truncate_store)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="Embed and index documents from the database.")

    parser.add_argument("--config", type=str, required=True, help="YAML config filename inside CONFIG_DIR.")
    parser.add_argument("--corpus_table", type=str, required=True, help="Name of the table containing raw documents.")
    parser.add_argument("--batch_size", type=int, default=500, help="Number of documents to process per batch.")
    parser.add_argument("--index_rebuild", action="store_true", help="Forces the index rebuild")
    parser.add_argument("--truncate_store", action="store_true", help="Truncates store table before indexing docs")


    args = parser.parse_args()
    main(args)


