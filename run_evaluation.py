import json
import argparse
from statistics import mean
from agents.evaluator import evaluate_single  # Ensure this import works

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

def main(input_file, output_file):
    all_scores = []

    for record in load_json_lines(input_file):
        prediction = record.get("prediction", "")
        references = record.get("references", [])
        target = record.get("target", "")
        source = record.get("input", "")

        if not references and target:
            references = [target]

        if not prediction or not references:
            continue

        score = evaluate_single(references, prediction, [source])
        all_scores.append(score)

    if not all_scores:
        print("No valid records for evaluation.")
        return

    # Compute average per metric
    average_scores = {
        "average_bleu": round(mean([s["BLEU"] for s in all_scores]), 3),
        "average_meteor": round(mean([s["METEOR"] for s in all_scores]), 3),
        "average_rouge_f1": round(mean([s["ROUGE-F1"] for s in all_scores]), 3),
        "comet_scores": round(mean([s["COMET"] for s in all_scores]), 3),
        "average_bertscore_f1": round(mean([s["BERTScore-F1"] for s in all_scores]), 3),
        "average_bleurt": round(mean([s["BLEURT"] for s in all_scores]), 3),
    }

    # Save to JSON file
    with open(output_file, "w") as f:
        json.dump(average_scores, f, indent=2)

    print(f"Saved average evaluation scores to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="Path to predictions JSON file")
    parser.add_argument("--output_file", required=True, help="Path to write averaged JSON file")
    args = parser.parse_args()

    main(args.input_file, args.output_file)
