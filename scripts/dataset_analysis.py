import json
import argparse
from statistics import mean
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

def plot_snippet_length_distribution(lengths, bins, labels, label_type="Snippet"):
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
    plt.title(f"{label_type} Length Distribution (in Tokens)")
    plt.xlabel("Token Length Intervals")
    plt.ylabel(f"Number of {label_type}s")
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

    def stats(label, lengths):
        print(f"{label} Statistics:")
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
        print()

    stats("Queries", query_lengths)
    stats("Snippets", snippet_lengths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze query and snippet lengths in a dataset.")
    parser.add_argument("--file", type=str, required=True, help="Path to JSON dataset.")
    parser.add_argument("--tokenizer", type=str, default="xlm-roberta-base", help="Tokenizer model name.")
    parser.add_argument("--use_char_count", action="store_true", help="Use character length instead of token length.")
    
    args = parser.parse_args()
    main(args)
