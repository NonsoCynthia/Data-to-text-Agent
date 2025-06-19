import os
import json
import argparse
from tqdm import tqdm
from agents.agents_modules.workflow import build_agent_workflow
from agents.dataloader import load_dataset_by_name, extract_example

# === CLI Arguments ===
parser = argparse.ArgumentParser()
parser.add_argument("--model_provider", required=True, help="Model provider (e.g., openai, ollama, hf, aixplain)")
parser.add_argument("--name", required=True, help="Dataset name (e.g., webnlg)")
parser.add_argument("--split", default="test", help="Dataset split (e.g., test)")
parser.add_argument("--output_file", required=True, help="Path to save predictions (.jsonl)")
parser.add_argument("--max_iteration", type=int, required=True, help="Agent max iteration count (e.g., 60)")
args = parser.parse_args()

# === Utility Function to Append Output ===
def append_to_file(output_file, output_data):
    with open(output_file, "a") as f:
        f.write(json.dumps(output_data) + "\n")

# === Load Existing Indices (if any) ===
completed_indices = set()
if os.path.exists(args.output_file):
    with open(args.output_file, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
                completed_indices.add(record["index"])
            except json.JSONDecodeError:
                continue  # skip corrupted lines

# === Initialize Agent Workflow ===
process_flow = build_agent_workflow(provider=args.model_provider)

# === Load Dataset ===
data = load_dataset_by_name(args.name)
dataset = data[args.split]
# print(f"Loaded {len(dataset)} examples from '{args.name}' [{args.split}]")
# print(f"Skipping {len(completed_indices)} already processed samples")

query = """You are an agent designed to generate text from data for a data-to-text natural language generation.
You may be provided data in XML, table, meaning representation, or graph format.
Your task is to generate fluent, complete text based strictly on the input.
Do not hallucinate or omit any facts.

Here is the data:
{input_data}"""

# === Run and Append Only Unseen Predictions ===
# use len(dataset) for full run
for i in tqdm(range(10), desc=f"Resumable run for {args.name}"):
    if i in completed_indices:
        continue

    sample = extract_example(args.name, dataset[i])
    input_data = sample.get("input", "")
    references = sample.get("references", [])
    target = sample.get("target", "")

    state = {
        "user_prompt": query.format(input_data=input_data),
        "raw_data": input_data,
        "history_of_steps": [],
        "final_response": "",
        "next_agent": "",
        "next_agent_payload": "",
        "current_step": 0,
        "iteration_count": 0,
        "max_iteration": args.max_iteration,
    }

    try:
        result = process_flow.invoke(state, config={"recursion_limit": args.max_iteration})
        prediction = result.get("final_response", "").strip()
    except Exception as e1:
        print(f"Failed at index {i}: {e1}")
        prediction = "ERROR"

    output = {
        "index": i,
        "prediction": prediction,
    }

    append_to_file(args.output_file, output)
