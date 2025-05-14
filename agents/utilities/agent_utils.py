import re
from typing import Dict, List, Text, Any, Union, Tuple
from agents.utilities.utils import StageExecute, ResultStep, ToolIntermediateStep


def validate_input_variables(template: Text, input_variables: Union[Text, Dict[Text, Text]]) -> Text:
    variables = re.findall(r"(?<!{){([^}]+)}(?!})", template)
    if isinstance(input_variables, str):
        return template
    for variable in variables:
        template = template.replace(f"{{{variable}}}", input_variables.get(variable, ""))
    return template


def prepare_tool_intermediate_steps(result_steps: list) -> List[Text]:
    try:
        tool_intermediate_steps = []
        for step, step_output in result_steps:
            tool_intermediate_steps.append(
                ToolIntermediateStep(
                    tool=step.tool,
                    input=step.tool_input,
                    output=step_output,
                )
            )
        return tool_intermediate_steps
    except Exception:
        return []

def prepare_result_steps(result_steps: List[ResultStep]) -> List[Text]:
    result_steps_str, nstep = [], 0
    for result_step in [step for step in result_steps if step.agent not in ["planner", "inspector"]]:
        if result_step.agent == "orchestrator":
            try:
                worker, worker_input = re.findall(r"(.*)\(input='(.*)'\)", result_step.output)[0]
            except Exception:
                worker, worker_input = "", ""
            if worker == "FINISH":
                result_steps_str.append(
                    f"***INTERMEDIATE STEP {nstep+1}:***\nWORKER: orchestrator\nWORKER RESPONSE:\n'{worker_input}'\n--------------------"
                )
                nstep += 1
        else:
            result_steps_str.append(
                f"***INTERMEDIATE STEP {nstep+1}:***\nWORKER: {result_step.agent}\nWORKER INPUT:\n'{result_step.input}'\nWORKER RESPONSE:\n'{result_step.output}'\n--------------------"
            )
            nstep += 1
    return result_steps_str

def render_intermediate_steps_xml(intermediate_steps: List[IntermediateStep]) -> str:
    """Render intermediate steps in XML format.

    Args:
        intermediate_steps: The intermediate steps to render.

    Returns:
        The rendered XML text.

    Output will be in the format of:

    .. code-block:: xml

        <intermediate_steps>
            <step number="1">
                <agent>agent_name</agent>
                <input>input_text</input>
                <response>output_text</response>
            </step>
            <step number="2">
                <agent>orchestrator</agent>
                <response>final_response</response>
            </step>
        </intermediate_steps>
    """
    if not intermediate_steps:
        return "<intermediate_steps></intermediate_steps>"

    # Filter out mentalist and inspector steps
    steps_without_mentalist_orchestrator_and_inspector = [
        step
        for step in intermediate_steps
        if "mentalist" not in step.agent
        and "orchestrator" not in step.agent
        and "inspector" not in step.agent
        and "feedback" not in step.agent
    ]

    step_strings = ["<intermediate_steps>"]

    for nstep, intermediate_step in enumerate(steps_without_mentalist_orchestrator_and_inspector, 1):
        step_strings.append(f'    <step number="{nstep}">')
        step_strings.append(f"        <agent>{intermediate_step.agent}</agent>")

        # Handle input and output, preserving None as empty string
        input_text = str(intermediate_step.input) if intermediate_step.input is not None else ""
        # TODO: Have a better way to handle this task parsing
        if input_text.startswith("Task:"):
            input_text = input_text.replace("Task:", "").split("\n\n")[0].strip()
        output_text = str(intermediate_step.output) if intermediate_step.output is not None else ""

        step_strings.append(f"        <input>{input_text}</input>")
        step_strings.append(f"        <response>{output_text}</response>")
        step_strings.append("    </step>")

    step_strings.append("</intermediate_steps>")
    return "\n".join(step_strings)


def render_chat_history_xml(chat_history: List[Dict[str, str]]) -> str:
    """Render chat history in a compact XML format.

    Args:
        chat_history: List of chat message dictionaries with 'role' and 'content' keys.

    Returns:
        The rendered XML text.

    Output will be in the format of:

    .. code-block:: xml

        <chat_history>
            <user>User message 1</user>
            <assistant>Assistant response 1</assistant>
            <user>User message 2</user>
            <assistant>Assistant response 2</assistant>
        </chat_history>
    """
    if not chat_history:
        return ""

    chat_strings = ["<chat_history>"]

    for message in chat_history:
        # Get role and content with empty string defaults
        role = message.get("role", "")
        content = message.get("content", "")

        # Convert None values to empty strings
        role = "" if role is None else role.upper()
        content = "" if content is None else str(content)

        chat_strings.append(f"    <{role}>{content}</{role}>")

    chat_strings.append("</chat_history>")
    return "\n".join(chat_strings)


def add_content_to_chat_history(chat_history: List[Dict[str, str]], content: Text, role: Text) -> List[Dict[str, str]]:
    """Add content to the chat history."""
    chat_history.append({"role": role, "content": content})
    return chat_history


def prepare_chat_history_xml(chat_history: List[Dict[str, str]]) -> str:
    """Format chat history for display using XML format.

    Args:
        chat_history: List of chat message dictionaries with 'role' and 'content' keys

    Returns:
        XML-formatted string representing the chat history, or empty string if no history
    """
    if not chat_history:
        return ""

    return render_chat_history_xml(chat_history)
