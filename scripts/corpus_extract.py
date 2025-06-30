import requests
import xml.etree.ElementTree as ET
import json
import time
from more_itertools import chunked
from db_connectors.postgres_connection import BasePostgreSQLConnectionHandler
from utils.app_config import BaseAppConfig
from db_connectors.base_connection import BaseConnectionHandler
from tqdm import tqdm
import argparse
import logging

def extract_unique_pmids(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    pmids = set()
    for q in data["questions"]:
        for doc_url in q["documents"]:
            pmid = doc_url.split("/")[-1]
            pmids.add(pmid)
    return list(pmids)


def get_missing_pmids(json_path, conn, table_name):
    json_pmids = extract_unique_pmids(json_path)
    existing_pmids = conn.get_existing_pmids(json_pmids, table_name)
    missing_pmids = [pmid for pmid in json_pmids if pmid not in existing_pmids]

    return missing_pmids


def extract_text(element):
    if isinstance(element, str):
        return element.strip()
    
    parts = []

    if element.text:
        parts.append(element.text)

    for child in element:
        parts.append(extract_text(child))
        if child.tail:
            parts.append(child.tail)

    return ''.join(parts).strip()


def parse_article(article_el, is_book=False):
    if is_book:
        base_el = article_el.find("BookDocument")
    else:
        base_el = article_el.find("MedlineCitation")

    if base_el is None:
        return

    is_available = True
    pmid = base_el.findtext("PMID")

    title_el = base_el.find(".//ArticleTitle")
    title = extract_text(title_el) if title_el is not None else None

    abstract_el = base_el.find(".//Abstract/AbstractText")
    abstract = extract_text(abstract_el) if abstract_el is not None else None

    if not abstract:
        is_available = False
    
    return pmid, title, abstract, is_available


def store_documents(pmids:list, db_conn: BaseConnectionHandler, ncbi_api_key: str, is_test: bool, batch_size: int) -> None:

    total_batches = (len(pmids) + batch_size - 1) // batch_size
    for pmid_batch in tqdm(chunked(pmids, batch_size), total=total_batches, desc="Storing documents"):
        response = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(pmid_batch),
                "api_key": ncbi_api_key,
                "retmode": "xml"
            }
        )

        time.sleep(0.1) # Just in case to avoid hitting the API rate limit.
        root = ET.fromstring(response.text)
        parsed_pmids = set()

        for record in root:
            article_data = None
            if record.tag == "PubmedArticle":
                article_data = parse_article(record, is_book=False)
            elif record.tag == "PubmedBookArticle":
                article_data = parse_article(record, is_book=True)
            else:
                parsed_pmids.add(int(article_data[0]))
                db_conn.store_document(*article_data)
                logging.warning(f"Pubmed article: {missing_pmid} content is not available.")
                continue

            db_conn.store_document(*article_data, is_test=is_test)
            parsed_pmids.add(int(article_data[0]))

        batch_pmids_int = set(map(int, pmid_batch))
        missing_pmids = batch_pmids_int - parsed_pmids

        for missing_pmid in missing_pmids:
            logging.warning(f"Pubmed article: {missing_pmid} is not available.")
            db_conn.store_document(
                missing_pmid,
                None,
                None,
                False,
                is_test=is_test
            )

def main(input_path: str, is_test: bool, batch_size: int) -> None:
    config = BaseAppConfig()
    corpus_table = config.corpus_table
    db_config = config.conn_config
    ncbi_api_key = config.ncbi_api_key
    db_conn = BasePostgreSQLConnectionHandler(db_config)

    db_conn.initialize_corpus_table(corpus_table)
    missing_pmids = get_missing_pmids(input_path, db_conn, corpus_table)
    clean_pmids = [pmid for pmid in missing_pmids if pmid.strip().isdigit()]
    if clean_pmids:
        store_documents(clean_pmids, db_conn, ncbi_api_key, is_test, batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and store PubMed documents.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--is_test", action="store_true", help="If the provided set of documents come from the evaluation files")
    parser.add_argument("--batch_size", type=int, default=300, help="Description of the new integer argument")
    args = parser.parse_args()

    main(args.input_path, args.is_test, args.batch_size)