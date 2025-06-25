import json
from datasets import Dataset
from ragas.evaluation import EvaluationResult
from ragas.metrics import answer_relevancy, faithfulness, context_precision
from ragas.llms import llm_factory
from ragas import evaluate
from dotenv import load_dotenv
from retrievers.base_retriever import BaseDocumentRetriever
from db_connectors.postgres_connection import BasePostgreSQLConnectionHandler
from utils.app_config import BaseAppConfig
from readers.base_reader import BaseReader
from readers.reader_factory import ChatModelFactory
import argparse
from rich import print
from rich.prompt import Prompt

class RagasEvaluator():
    """
    Evaluator class to wrap all required objects and configuration to perform the evaluation over
    the QA pipeline

    Attributes:

    retriever: BaseDocumentRetriever that interacts with the vector store
    reader: BaseReader class object implementing the interface, allows interaction with the LLM that generates the answer
    search_top_k: top k docs to retrieve from the vector store
    reranker_top_k: top k docs to get after reranking the documents fetched from the vector store
    llm_top_k: top k docs to feed to the llm within the context.
    use_reranker: bool flag to activate the reranking in the evaluation pipeline.
    fts_search: bool flag to set the type of search (dense retrieval with emebddins or ful text search based on term frequency)

    """
    def __init__(self, retriever: BaseDocumentRetriever, reader: BaseReader, search_top_k: int=100, llm_top_k: int=10, use_reranker:bool=True, reranker_top_k:int=10, fts_search: bool=False):
        self.retriever = retriever
        self.reader = reader
        self.search_top_k = search_top_k
        self.reranker_top_k = reranker_top_k
        self.llm_top_k = llm_top_k
        self.use_reranker = use_reranker
        self.fts_search = fts_search
    
    def create_ragas_dataset(self, evaluation_file_path: str):
        """
        From the evaluation file provided triggers the retrieval pipeline and generates an awnswer based on the
        provided configuration. All records are merged into a python list used to create the Dataset object for the evaluation
        function used from RAGAS
        """

        with open(evaluation_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)["questions"]

        ragas_records = []

        for record in data:
            question = record["body"]
            ground_truths = record.get("ideal_answer", []) or record.get("exact_answer", [])
            ground_truth = ground_truths[0] if ground_truths else ""

            if not self.fts_search:
                docs = self.retriever.get_top_k_docs(question, self.search_top_k)
            else:
                docs = self.retriever.get_top_k_docs_fts(question, search_top_k=self.search_top_k)
            
            if self.use_reranker:
                docs = self.retriever.rerank_top_k_docs(question, docs, self.reranker_top_k)
            contexts = [doc["content"] for doc in docs[:self.llm_top_k] if "content" in doc]

            answer = self.reader.answer(question, docs[:self.llm_top_k])

            ragas_records.append({
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth
            })

        return Dataset.from_list(ragas_records)

    def run_ragas_evaluation(self, dataset: Dataset, evaluator_llm_name: str) -> EvaluationResult:
        """
        From the evaluation file provided triggers the retrieval pipeline and generates an awnswer based on the
        provided configuration. All records are merged into a python list used to create the Dataset object for the evaluation
        function used from RAGAS module

        Return a EvaluationResult object
        """
        result = evaluate(
            dataset,
            metrics=[
                answer_relevancy,
                faithfulness,
                context_precision
            ],
            llm=llm_factory(model=evaluator_llm_name)
        )
        return result

def main(args):
    load_dotenv()
    config = BaseAppConfig()
    evaluation_file_path = args.evaluation_file_path
    search_top_k = args.search_top_k
    rerank_top_k = args.rerank_top_k
    llm_top_k = args.llm_top_k
    use_reranker = args.use_reranker
    search_mode = args.search_mode

    # Get Embedder to be used
    embedder_categories = list(config.embedder_config.keys())
    print(f"\nAvailable embedder categories: {embedder_categories}")
    selected_category = Prompt.ask("Choose a category", choices=embedder_categories)

    category_embedders = config.embedder_config[selected_category]
    embedder_names = [e["name"] for e in category_embedders]
    print(f"\nAvailable embedders in [{selected_category}]: {embedder_names}")
    selected_embedder_name = Prompt.ask("Choose an embedder", choices=embedder_names)
    selected_embedder = next(e for e in category_embedders if e["name"] == selected_embedder_name)

    # Get Reader to be used
    reader_categories = list(config.reader_config.keys())
    print(f"\nAvailable embedder categories: {reader_categories}")
    selected_reader_category = Prompt.ask("Choose a category", choices=reader_categories)

    category_readers = config.reader_config[selected_reader_category]
    reader_names = [e["name"] for e in category_readers]
    print(f"\nAvailable readers in [{selected_category}]: {reader_names}")
    selected__reader_name= Prompt.ask("Choose an reader", choices=reader_names)
    selected_reader = next(e for e in category_readers if e["name"] == selected__reader_name)

    if use_reranker:
        # Get Reranker to be used
        reranker_categories = list(config.reranker_config.keys())
        print(f"\nAvailable reranker categories: {reranker_categories}")
        selected_reranker_category = Prompt.ask("Choose a reranker category", choices=reranker_categories)
        category_rerankers = config.reranker_config[selected_reranker_category]

        reranker_names = [r["name"] for r in category_rerankers]
        print(f"\nAvailable rerankers in [{selected_reranker_category}]: {reranker_names}")
        selected_reranker_name = Prompt.ask("Choose a reranker", choices=reranker_names)
        selected_reranker = next(e for e in category_rerankers if e["name"] == selected_reranker_name)

    retriever = BaseDocumentRetriever(
        store_handler=BasePostgreSQLConnectionHandler(config.conn_config),
        embedding_config= selected_embedder.get("model_config") if selected_embedder_name != "fts" else None,
        reranker_config= None if not use_reranker else selected_reranker,
        corpus_table="documents",
        store_table=selected_embedder.get("store_table")
    )
    reader = ChatModelFactory.get_model(
                                    """Answer the following question as precisely and concisely as possible. 
                                                    Do not include context explanations, reasoning, or observations. 
                                                    Respond with only the direct answer, in a single sentence or a short list if needed.
                                                    If the context information is not enough to elaborate an answer, just say you don't know the answer.""",
                                        selected_reader_category,
                                        model_config_kargs=selected_reader.get("model_config")
                                    )
    
    
    evaluator = RagasEvaluator(retriever, reader,search_top_k=search_top_k, use_reranker=use_reranker, reranker_top_k=rerank_top_k, llm_top_k=llm_top_k, search_mode=search_mode)
    evaluator_llm = llm_factory(model="gpt-4o") # If not set gtp-4o mini used by default
    ragas_dataset = evaluator.create_ragas_dataset(evaluation_file_path)
    eval_results = evaluator.run_ragas_evaluation(ragas_dataset, evaluator_llm)
    print(eval_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG evaluation with configurable parameters.")
    parser.add_argument("--evaluation_file_path", type=str, default="data/test/evaluation_files/12B1_golden.json", help="Path to the evaluation JSON file.")
    parser.add_argument("--search_top_k", type=int, default=100, help="Number of documents to retrieve from retriever.")
    parser.add_argument("--rerank_top_k", type=int, default=10, help="Number of documents to rerank.")
    parser.add_argument("--llm_top_k", type=int, default=10, help="Number of documents to pass to the LLM reader.")
    parser.add_argument("--use_reranker", action="store_true", help="Whether to use reranker after retrieval.")
    parser.add_argument("--search_mode", type=str, default="embedding", choices=["embedding", "fts"], help="Search mode: embedding or full text search.")
    main()