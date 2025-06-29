import json
import argparse
from statistics import mean
# from agents.evaluator import evaluate_single
from agents.evaluator import BatchEvaluator  # NEW
from agents.dataloader import load_dataset_by_name, extract_example


def load_json_lines(path):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def main(input_file, dataset_name, split, output_file):
    dataset = load_dataset_by_name(dataset_name)[split]
    evaluator = BatchEvaluator()  # load checkpoints
    scores = []
    avg = {}

    for rec in load_json_lines(input_file):
        idx = rec.get("index")
        pred = rec.get("prediction", "")
        if idx is None or not pred:
            continue

        sample = extract_example(dataset_name, dataset[idx])
        refs = sample.get("references") or [sample.get("target", "")]
        src = sample.get("input", "")
        if not refs or not pred:
            continue

        scores.append(evaluator.score(refs, pred, [src]))

    if not scores:
        print("No valid records for evaluation.")
        return
    

    avg = {
        "average_bleu": round(mean(s["BLEU"] for s in scores), 3),
        "average_meteor": round(mean(s["METEOR"] for s in scores), 3),
        "average_rouge_f1": round(mean(s["ROUGE-F1"] for s in scores), 3),
        "average_comet": round(mean(s["COMET"] for s in scores), 3),
        "average_bertscore_f1": round(mean(s["BERTScore-F1"] for s in scores), 3),
        "average_bleurt": round(mean(s["BLEURT"] for s in scores), 3),
    }

    # avg["average_bleu"] = round(mean(s["BLEU"] for s in scores), 3)
    # avg["average_meteor"] = round(mean(s["METEOR"] for s in scores), 3)
    # avg["average_rouge_f1"] = round(mean(s["ROUGE-F1"] for s in scores), 3)
    # avg["average_comet"] = round(mean(s["COMET"] for s in scores), 3)
    # avg["average_bertscore_f1"] = round(mean(s["BERTScore-F1"] for s in scores), 3)
    # avg["average_bleurt"] = round(mean(s["BLEURT"] for s in scores), 3)

    with open(output_file, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(avg) + "\n")

    print(f"Saved average evaluation scores to {output_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_file", required=True)
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--dataset_split", default="test")
    p.add_argument("--output_file", required=True)
    args = p.parse_args()
    main(args.input_file, args.dataset_name, args.dataset_split, args.output_file)
