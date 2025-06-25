import json
import argparse


def prepare_finetune_data(bioasq_json_path, output_path, prompt):
    with open(bioasq_json_path) as f:
        data = json.load(f)

    output = []

    for q in data["questions"]:
        question = q["body"]
        snippets = q.get("snippets", [])
        positives = []

        for s in snippets:
            text = s.get("text", "").strip()
            if text:
                positives.append(text)

        if positives:
            entry = {
                "query": question,
                "pos": positives,
                "neg": [],
                "prompt": prompt,
                "type": "normal"
            }
            output.append(entry)

    with open(output_path, "w") as fout:
        for ex in output:
            fout.write(json.dumps(ex) + "\n")

    print(f"Saved {len(output)} entries to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Creates input file for fine tunnign process from BioASQ train file.")
    parser.add_argument("--bioasq_json_path", type=str, required=True, help="Path to the BioASQ train JSON file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output JSONL file")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to include before each question")

    args = parser.parse_args()

    prepare_finetune_data(args.bioasq_json_path, args.output_path, args.prompt)

if __name__ == "__main__":
    main()

