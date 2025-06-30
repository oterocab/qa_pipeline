import pandas as pd
from statistics import mean
import math
import json


def match_snippet_in_results(snippet, retrieved_docs, overlap_threshold=0.5):
    snippet_doc_id = snippet["document"].split("/")[-1]
    snippet_start = snippet["offsetInBeginSection"]
    snippet_end = snippet["offsetInEndSection"]
    snippet_len = snippet_end - snippet_start
    snippet_section = snippet.get("beginSection")

    best_match = None
    best_overlap = 0

    for rank, doc in enumerate(retrieved_docs):
        doc_id = doc.get("id")
        if doc_id != snippet_doc_id:
            continue

        doc_start = doc.get("beginoffset")
        doc_end = doc.get("endoffset")
        doc_chunk = doc.get("chunk")
        doc_section = "title" if doc_chunk == 0 else "abstract"

        if snippet_section != doc_section:
            continue

        overlap_start = max(snippet_start, doc_start)
        overlap_end = min(snippet_end, doc_end)
        overlap_len = max(0, overlap_end - overlap_start)
        overlap_ratio = overlap_len / snippet_len if snippet_len > 0 else 0

        if overlap_ratio > best_overlap:
            best_overlap = overlap_ratio
            best_match = rank

    return best_match if best_overlap >= overlap_threshold else None


def compute_metrics_from_ranks(ranks, k, total_relevant):
    if not ranks:
        return {
            f"MRR@{k}": 0,
            f"Recall@{k}": 0,
            f"Precision@{k}": 0,
            f"F1@{k}": 0,
            f"nDCG@{k}": 0
        }

    hits_at_k = [r for r in ranks if r < k]
    num_hits = len(hits_at_k)

    mrr = 1 / (min(hits_at_k) + 1) if hits_at_k else 0
    precision = num_hits / k
    recall = num_hits / total_relevant if total_relevant > 0 else 0
    dcg = sum(1 / math.log2(r + 2) for r in hits_at_k)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(total_relevant, k)))
    ndcg = dcg / idcg if idcg > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return {
        f"MRR@{k}": mrr,
        f"Recall@{k}": recall,
        f"Precision@{k}": precision,
        f"F1@{k}": f1,
        f"nDCG@{k}": ndcg
    }

def evaluate_retriever_from_evaldata(retriever, eval_file_path: str, ks=(5, 10, 15), top_k=100, 
                                     rerank_top_k=30, use_rerank=True, overlap_threshold=0.5, 
                                     fts_search: bool = False, max_workers: int = 1):
    
    results = []

    with open(eval_file_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    questions = eval_data["questions"]

    queries = [q["body"] for q in questions]
    snippets_list = [q.get("snippets", []) for q in questions]

    batch_docs, batch_retrieval_latencies = retriever.get_top_k_docs_batch(queries=queries, top_k=top_k, fts_search=fts_search, max_workers=max_workers)
    batch_docs = [docs if docs is not None else [] for docs in batch_docs]
    mean_rerank_lantecy = 0
    if use_rerank:
        rerank_results = [
            retriever.rerank_top_k_docs(query, docs, rerank_top_k)
            for query, docs in zip(queries, batch_docs)
        ]
        batch_docs = [result[0] for result in rerank_results]
        batch_rerank_latencies = [result[1] for result in rerank_results]
        mean_rerank_lantecy = mean(batch_rerank_latencies)

    metric_names = [f"{metric}@{k}"for k in ks
                        for metric in ("MRR", "Recall", "Precision", "F1", "nDCG")]
    for query, snippets, docs in zip(queries, snippets_list, batch_docs):
        query_result = {
            "query": query,
            "total_snippets": len(snippets),
            **{m: 0 for m in metric_names},
            #**{f"hits@{k}": 0 for k in ks}
        }
        available_ks = [k for k in ks if k <= len(docs)]
        for k in available_ks:
            top_k_docs = docs[:k]
            ranks = []
            used_doc_ids = set()

            for snip in snippets:
                rank = match_snippet_in_results(snip, top_k_docs, overlap_threshold=overlap_threshold)
                if rank is not None:
                    doc_id = top_k_docs[rank].get("id", "")
                    if doc_id not in used_doc_ids:
                        ranks.append(rank)
                        used_doc_ids.add(doc_id)

            metrics = compute_metrics_from_ranks(ranks, k, len(snippets))
            for metric_name, value in metrics.items():
                query_result[metric_name] = value

            #query_result[f"hits@{k}"] = len(ranks)

        results.append(query_result)

    return pd.DataFrame(results), mean(batch_retrieval_latencies), mean_rerank_lantecy