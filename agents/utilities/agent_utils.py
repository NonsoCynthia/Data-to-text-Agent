import re
import torch
from typing import Dict, List, Text, Any, Union, Optional
from comet import download_model, load_from_checkpoint
from agents.utilities.utils import ExecutionState, AgentStepOutput, IntermediateToolUsage

# Configure precision
torch.set_float32_matmul_precision("high")

_comet_instance = None

def apply_variable_substitution(template: Text, substitutions: Union[Text, Dict[Text, Text]]) -> Text:
    keys = re.findall(r"(?<!{){([^}]+)}(?!})", template)
    if isinstance(substitutions, str):
        return template
    for key in keys:
        template = template.replace(f"{{{key}}}", substitutions.get(key, ""))
    return template

def summarize_agent_steps(step_log: List[AgentStepOutput]) -> List[Text]:
    summary, idx = [], 0
    for entry in [s for s in step_log if s.agent_name not in ["planner", "inspector"]]:
        if entry.agent_name == "orchestrator":
            try:
                role, input_payload = re.findall(r"(.*)\(input='(.*)'\)", entry.agent_output)[0]
            except Exception:
                role, input_payload = "", ""
            if role == "FINISH":
                summary.append(
                    f"***STEP {idx+1}:***\nAGENT: orchestrator\nRESPONSE:\n'{input_payload}'\n--------------------"
                )
                idx += 1
        else:
            summary.append(
                f"***STEP {idx+1}:***\nAGENT: {entry.agent_name}\nINPUT:\n'{entry.agent_input}'\nRESPONSE:\n'{entry.agent_output}'\n--------------------"
            )
            idx += 1
    return summary

def to_result_steps_xml(agent_steps: List[AgentStepOutput]) -> str:
    if not agent_steps:
        return "<result_steps></result_steps>"
    filtered = [step for step in agent_steps if all(x not in step.agent_name for x in ["planner", "orchestrator", "inspector"])]
    xml_output = ["<result_steps>"]
    for count, step in enumerate(filtered, 1):
        user_input = str(step.agent_input or "")
        if user_input.startswith("Task:"):
            user_input = user_input.replace("Task:", "").split("\n\n")[0].strip()
        agent_response = str(step.agent_output or "")
        xml_output += [
            f'    <step number="{count}">',
            f"        <agent>{step.agent_name}</agent>",
            f"        <input>{user_input}</input>",
            f"        <response>{agent_response}</response>",
            "    </step>"
        ]
    xml_output.append("</result_steps>")
    return "\n".join(xml_output)

def score_comet_quality(source_text: str, prediction_text: str, use_gpu=False):
    global _comet_instance

    if _comet_instance is None:
        model_path = download_model("Unbabel/wmt22-cometkiwi-da")
        _comet_instance = load_from_checkpoint(model_path)

    torch.cuda.empty_cache()
    run_on_gpu = 1 if use_gpu and torch.cuda.is_available() else 0

    source = str(source_text).strip()
    mt = str(prediction_text).strip()

    try:
        result = _comet_instance.predict(
            [{'src': source, 'mt': mt}],
            gpus=run_on_gpu,
            num_workers=1  # avoids multiprocessing_context error
        )
        return f"Metric Evaluation Result: {round(result.system_score, 3)}"
    except Exception as e:
        print(f"Metric Evaluation Result: Metric Evaluation Failed: {e}")
        return ""


def find_validated_agents(step_trace: List[AgentStepOutput]) -> set:
    verified = set()
    for i, step in enumerate(step_trace):
        if step.agent_name == "inspector" and str(step.agent_output).strip().upper() == "CORRECT":
            for j in range(i - 1, -1, -1):
                prior = step_trace[j]
                if prior.agent_name not in ["inspector", "orchestrator", "planner"]:
                    verified.add(prior.agent_name.lower())
                    break
    return verified
