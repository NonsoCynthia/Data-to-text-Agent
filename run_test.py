import os
import argparse
from tqdm import tqdm
from datetime import datetime
from agents.agents_modules.workflow import build_agent_workflow
from agents.dataloader import load_dataset_by_name, extract_example

# === CLI Arguments ===
parser = argparse.ArgumentParser()
parser.add_argument("--model_provider", required=True, help="Model provider (e.g., openai, ollama, hf, aixplain)")
parser.add_argument("--name", required=True, help="Dataset name (e.g., webnlg)")
parser.add_argument("--split", default="test", help="Dataset split (e.g., test)")
parser.add_argument("--output_file", required=True, help="Path to save all predictions (.txt)")
parser.add_argument("--max_iteration", required=True, help="Agent max iteration count (e.g., 60)")
args = parser.parse_args()

# === Initialize Agent Workflow ===
process_flow = build_agent_workflow(provider=args.model_provider)

# === Load Dataset ===
data = load_dataset_by_name(args.name)
dataset = data[args.split]
print(f"Loaded {len(dataset)} examples from '{args.name}' [{args.split}]")

# === Generate and Save Predictions ===
with open(args.output_file, "w") as f:
    for i in tqdm(range(len(dataset)), desc=f"Generating predictions for {args.name}"):
        sample = extract_example(args.name, dataset[i])
        input_data = sample.get("input", "")

        query = f"""You are an agent designed to generate text from data for a data-to-text natural language generation.
You may be provided data in XML, table, meaning representation, or graph format.
Your task is to generate fluent, complete text based strictly on the input.
Do not hallucinate or omit any facts.

Here is the data:
{input_data}
"""

        state = {
            "user_prompt": query,
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
            result = process_flow.invoke(state, config={"recursion_limit": 60})
        except Exception as e1:
            print(f"Attempt 1 failed at index {i}: {e1}")
            try:
                result = process_flow.invoke(state, config={"recursion_limit": 60})
            except Exception as e2:
                print(f"Attempt 2 failed at index {i}: {e2}")
                f.write("[ERROR]\n")
                continue

        prediction = result.get("final_response", "").strip()
        f.write(prediction + "\n")
