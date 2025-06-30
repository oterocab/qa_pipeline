import json
import argparse
from statistics import mean
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

def plot_snippet_length_distribution(lengths, bins, labels, label_type="Snippet", count_by: str = "Token"):
    counts = [0] * len(labels)

    for length in lengths:
        placed = False
        for i, upper in enumerate(bins):
            if length <= upper:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1

    plt.figure(figsize=(8, 5))
    plt.bar(labels, counts, width=0.6)
    plt.title(f"Distribución de longitud de {label_type} (en {count_by}s)")
    plt.xlabel(f"Intervalos de longitud en {count_by}")
    plt.ylabel(f"Número de {label_type}s")
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_lengths(texts, tokenizer=None, use_char_count=False):
    if use_char_count:
        return [len(text) for text in texts]
    else:
        return [len(tokenizer.encode(text, truncation=False)) for text in texts]


def main(args):
    data = load_data(args.file)

    queries = [q["body"] for q in data["questions"]]
    snippets = [s["text"] for q in data["questions"] for s in q.get("snippets", [])]

    tokenizer = None
    if not args.use_char_count:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    query_lengths = compute_lengths(queries, tokenizer, args.use_char_count)
    snippet_lengths = compute_lengths(snippets, tokenizer, args.use_char_count)
    snippet_counts = [len(query.get("snippets", [])) for query in data["questions"]]

    def stats(label, lengths):
        print(f"{label} Estadísticas:")
        print(f"Count: {len(lengths)}")
        print(f"Min: {min(lengths)}")
        print(f"Max: {max(lengths)}")
        print(f"Avg: {mean(lengths):.2f}")
        if not args.use_char_count:
            token_bins = [64, 128, 256, 512]
            bin_labels = ["≤64", "65–128", "129–256", "257–512", ">512"]
            plot_snippet_length_distribution(lengths, token_bins, bin_labels, label_type=label)
            over_limit = sum(1 for l in lengths if l > tokenizer.model_max_length)
            print(f"> {tokenizer.model_max_length} tokens: {over_limit} ({100 * over_limit / len(lengths):.2f}%)")
        else:
            char_bins = [64, 128, 256, 512, 1024]
            bin_labels = ["≤64", "65–128", "129–256", "257–512", "513-1024", ">1024"]
            plot_snippet_length_distribution(lengths, char_bins, bin_labels, label_type=label, count_by="Character")

    bins_labels = ["1-5", "5-10", "10-20", "20-50", "50+"]
    bins_upper = [5, 10, 20, 50]
    counts_hist = [0] * len(bins_labels)
    for c in snippet_counts:
        placed = False
        for i, upper in enumerate(bins_upper):
            if c <= upper:
                counts_hist[i] += 1
                placed = True
                break
        if not placed:
            counts_hist[-1] += 1
    plt.figure(figsize=(8, 5))
    plt.bar(bins_labels, counts_hist)
    plt.title("Número de Snippets")
    plt.ylabel("Número de Queries")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

    stats("Queries", query_lengths)
    stats("Snippets", snippet_lengths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze query and snippet lengths in a dataset.")
    parser.add_argument("--file", type=str, required=True, help="Path to JSON dataset.")
    parser.add_argument("--tokenizer", type=str, default="xlm-roberta-base", help="Tokenizer model name.")
    parser.add_argument("--use_char_count", action="store_true", help="Use character length instead of token length.")
    
    args = parser.parse_args()
    main(args)
