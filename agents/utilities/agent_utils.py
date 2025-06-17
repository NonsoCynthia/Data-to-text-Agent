import os
import re
import json
from typing import List, Text, Union, Dict
from agents.utilities.utils import AgentStepOutput

def apply_variable_substitution(template: Text, substitutions: Union[Text, Dict[Text, Text]]) -> Text:
    """
    Substitute placeholders in the form {variable} within a template string.

    Args:
        template (Text): A string containing placeholders.
        substitutions (Union[Text, Dict[Text, Text]]): Substitution values.

    Returns:
        Text: The formatted string with placeholders replaced.
    """
    if isinstance(substitutions, str):
        return template  # no substitution possible with plain string

    pattern = re.compile(r"(?<!{){([^{}]+)}(?!})")
    for match in pattern.findall(template):
        template = template.replace(f"{{{match}}}", substitutions.get(match, ""))
    return template

def summarize_agent_steps(step_log: List[AgentStepOutput]) -> List[Text]:
    """
    Generate a uniquely formatted summary of agent execution steps,
    excluding guardrail steps and using UID-tagged blocks.

    Args:
        step_log (List[AgentStepOutput]): List of agent interaction records.

    Returns:
        List[Text]: A list of UID-formatted step summaries.
    """
    summary = []
    step_counter = 1

    for entry in step_log:
        agent = entry.agent_name.lower()

        if agent == "guardrail":
            continue

        if agent == "orchestrator":
            try:
                role, role_input = re.findall(r"(.*)\(input='(.*)'\)", entry.agent_output)[0]
            except Exception:
                role, role_input = "FINISH", entry.agent_output

            agent_type = "orchestrator"
            uid = f"{agent_type.upper()}_{step_counter}"
            if role == "FINISH":
                block = (
                    f"##=== BEGIN:{uid} ===##\n"
                    f"-- AGENT TYPE: {agent_type}\n"
                    f"-- AGENT NAME: {entry.agent_name}\n"
                    f"-- SIGNAL: FINISH\n"
                    f"-- RESPONSE START --\n{role_input}\n-- RESPONSE END --\n"
                    f"##=== END:{uid} ===##"
                )
            else:
                block = (
                    f"##=== BEGIN:{uid} ===##\n"
                    f"-- AGENT TYPE: {agent_type}\n"
                    f"-- AGENT NAME: {entry.agent_name}\n"
                    f"-- ROUTED TO: {role}\n"
                    f"-- INPUT START --\n{role_input}\n-- INPUT END --\n"
                    f"##=== END:{uid} ===##"
                )

        else:
            agent_type = "worker"
            uid = f"{agent_type.upper()}_{step_counter}"
            block = (
                f"##=== BEGIN:{uid} ===##\n"
                f"-- AGENT TYPE: {agent_type}\n"
                f"-- AGENT NAME: {entry.agent_name}\n"
                f"-- INPUT START --\n{entry.agent_input}\n-- INPUT END --\n"
                f"-- OUTPUT START --\n{entry.agent_output}\n-- OUTPUT END --\n"
                f"##=== END:{uid} ===##"
            )

        summary.append(block)
        step_counter += 1

    return summary


def save_result_to_json(state: dict, filename: str = "result.json", directory: str = "results") -> None:
    """
    Saves the given agent workflow state to a JSON file in a specified directory.

    Args:
        state (dict): The final state returned by the agent graph execution.
        filename (str): The name of the JSON file to save.
        directory (str): The directory where the file will be saved.

    Raises:
        IsADirectoryError: If the target file path is a directory.
    """

    file_path = os.path.join(directory, filename)
    os.makedirs(directory, exist_ok=True)

    if os.path.isdir(file_path):
        raise IsADirectoryError(f"Cannot write to '{file_path}' because it is a directory.")

    # Recursively convert objects to serializable types
    def make_serializable(obj):
        if isinstance(obj, list):
            return [make_serializable(x) for x in obj]
        elif hasattr(obj, "model_dump"):  # Pydantic BaseModel
            return obj.model_dump()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        else:
            return obj

    serializable_state = make_serializable(dict(state))

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(serializable_state, f, indent=4)

    print(f"Results saved to {file_path}")


# import torch
# from comet import download_model, load_from_checkpoint
# Configure precision
# torch.set_float32_matmul_precision("high")
# _comet_instance = None

# def score_comet_quality(source_text: str, prediction_text: str, use_gpu=False):
#     global _comet_instance

#     if _comet_instance is None:
#         model_path = download_model("Unbabel/wmt22-cometkiwi-da")
#         _comet_instance = load_from_checkpoint(model_path)

#     torch.cuda.empty_cache()
#     run_on_gpu = 1 if use_gpu and torch.cuda.is_available() else 0

#     source = str(source_text).strip()
#     mt = str(prediction_text).strip()

#     try:
#         result = _comet_instance.predict(
#             [{'src': source, 'mt': mt}],
#             gpus=run_on_gpu,
#             num_workers=1  # avoids multiprocessing_context error
#         )
#         return f"Metric Evaluation Result: {round(result.system_score, 3)}"
#     except Exception as e:
#         print(f"Metric Evaluation Result: Metric Evaluation Failed: {e}")
#         return ""


