import os
import json
import argparse
from tqdm import tqdm
from agents.agents_modules.workflow import build_agent_workflow
from agents.dataloader import load_dataset_by_name, extract_example
from agents.utilities.agent_utils import save_result_to_json
from agents.llm_model import UnifiedModel, model_name
from agents.agent_prompts import END_TO_END_GENERATION_PROMPT, input_prompt

def build_d2t_prompt(name, num_examples, input_data, input_prompt, samples_file="random_train_samples.json"):
    with open(samples_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    filtered = [
        ex for ex in results
        if ex.get("dataset") == name and ex.get("input", "")
        and (ex.get("target", "").strip() or (ex.get("references") and any(r.strip() for r in ex["references"])))
    ]
    examples = filtered[:num_examples]
    if len(examples) < num_examples:
        print(f"Only {len(examples)} examples found for dataset '{name}'")
    prompt_blocks = []
    for i, ex in enumerate(examples, 1):
        input_text = ex.get("input", "")
        output = ex.get("target", "")
        if not output:
            refs = [r.strip() for r in ex.get("references", []) if r.strip()]
            output = refs[0] if refs else ""
        prompt_blocks.append(f"Example {i}:\nData: {input_text}\nOutput: {output}\n")
    prompt_examples = "\n".join(prompt_blocks)
    prompt_template = input_prompt.format(dataset_name=name,
                        examples=f"\n\nExamples: {prompt_examples}\n\n",
                        data=input_data )
    return prompt_template


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_provider", required=True, help="Model provider (e.g., openai, ollama, hf, aixplain)")
    parser.add_argument("--name", required=True, help="Dataset name (e.g., webnlg)")
    parser.add_argument("--split", default="test", help="Dataset split (e.g., test)")
    parser.add_argument("--type", default="test", help="Generation type: 'agent' or 'e2e'")
    parser.add_argument("--output_file", required=True, help="Path to save predictions (output.jsonl)")
    parser.add_argument("--max_iteration", type=int, required=True, help="Max iteration count for agent execution")
    return parser.parse_args()

def append_to_file(output_file, output_data):
    with open(output_file, "a") as f:
        f.write(json.dumps(output_data) + "\n")

def load_completed_indices(output_file):
    indices = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    indices.add(record["index"])
                except json.JSONDecodeError:
                    continue
    return indices

def run():
    args = parse_args()
    completed_indices = load_completed_indices(args.output_file)
    exg_name = 'webnlg' if args.name == 'webnlg_hf' else args.name
    dataset = load_dataset_by_name(args.name)[args.split]
    workflow = build_agent_workflow(provider=args.model_provider)
    conf = model_name.get(args.model_provider.lower(), {})
    conf["temperature"] = 0.0
    
    num_samples = len(dataset)
    print(f"Processing {num_samples} samples from '{args.name}' ({args.split})...")

    for i in tqdm(range(num_samples), desc=f"{args.type.upper()} Generation"):
        if i in completed_indices:
            continue
        sample = extract_example(args.name, dataset[i])
        input_data = sample.get("input", "")
        # print(input_data)
        # references = sample.get("references", [])
        # target = sample.get("target", "")

        if args.type == "agent":
            # llm = UnifiedModel(provider=args.model_provider, **conf).model_(CONTENT_SELECTION_PROMPT)
            # content_extract = llm.invoke({'input': input_data}).content.strip()
            # print(content_extract)
            prompt = input_prompt.format(dataset_name=exg_name, data=input_data)
        else:
            prompt = build_d2t_prompt(name=exg_name, num_examples=5, input_data=input_data, input_prompt=input_prompt)
            # print(prompt)

        try:
            if args.type == "agent":
                # Agent-based generation
                state = {
                    "user_prompt": prompt,
                    "max_iteration": args.max_iteration,
                }
                result = workflow.invoke(state, config={"recursion_limit": args.max_iteration})
                # print(result)
                if result:
                    save_result_to_json(result, dataset_folder=f"{exg_name}", filename=f"{exg_name}_{i}.json")
                    prediction = result.get("final_response", "").strip()
                else:
                    print(f"[WARNING] Empty state returned at index {i}")

            else:
                # End-to-end generation
                llm = UnifiedModel(provider=args.model_provider, **conf).model_(END_TO_END_GENERATION_PROMPT)
                prediction = llm.invoke({'input': prompt}).content.strip()

        except Exception as err:
            print(f"[Error] Index {i}: {err}")
            prediction = "ERROR"

        append_to_file(args.output_file, {
            "index": i,
            "prediction": prediction,
            # "input": input_data,        # optionally include these
            # "references": references,
            # "target": target
        })

if __name__ == "__main__":
    run()
