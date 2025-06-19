import json
import argparse
from statistics import mean
from agents.evaluator import evaluate_single
from agents.dataloader import load_dataset_by_name, extract_example

def load_json_lines(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # skip empty lines
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def main(input_file, dataset_name, dataset_split, output_file):
    all_scores = []

    # Load the dataset once
    dataset = load_dataset_by_name(dataset_name)[dataset_split]

    for record in load_json_lines(input_file):
        index = record.get("index")
        prediction = record.get("prediction", "")

        if index is None or not prediction:
            continue

        # Fetch references and input from dataset
        sample = extract_example(dataset_name, dataset[index])
        references = sample.get("references", [])
        target = sample.get("target", "")
        source = sample.get("input", "")

        if not references and target:
            references = [target]

        if not references:
            continue

        score = evaluate_single(references, prediction, [source])
        all_scores.append(score)

    if not all_scores:
        print("No valid records for evaluation.")
        return

    average_scores = {
        "average_bleu": round(mean([s["BLEU"] for s in all_scores]), 3),
        "average_meteor": round(mean([s["METEOR"] for s in all_scores]), 3),
        "average_rouge_f1": round(mean([s["ROUGE-F1"] for s in all_scores]), 3),
        "comet_scores": round(mean([s["COMET"] for s in all_scores]), 3),
        "average_bertscore_f1": round(mean([s["BERTScore-F1"] for s in all_scores]), 3),
        "average_bleurt": round(mean([s["BLEURT"] for s in all_scores]), 3),
    }

    with open(output_file, "w") as f:
        json.dump(average_scores, f, indent=2)

    print(f"Saved average evaluation scores to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_split", default="test")
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    main(args.input_file, args.dataset_name, args.dataset_split, args.output_file)
