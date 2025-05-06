import re
from typing import Dict, List, Text, Any, Union, Tuple
from langchain.agents import AgentExecutor, create_json_chat_agent
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from utils import StageExecute, ResultStep, ToolIntermediateStep
from llm_model import OllamaModel
from agent_prompts import WORKER_PROMPT, AGENT_SYSTEM_PROMPT, AGENT_HUMAN_PROMPT
from langgraph.errors import GraphRecursionError


def validate_input_variables(template: Text, input_variables: Union[Text, Dict[Text, Text]]) -> Text:
    """
    Replaces variables within the template using values from input_variables.
    If input_variables is a string, the template is returned unchanged.
    """
    variables = re.findall(r"(?<!{){([^}]+)}(?!})", template)
    if isinstance(input_variables, str):
        return template
    for variable in variables:
        template = template.replace(f"{{{variable}}}", input_variables.get(variable, ""))
    return template


def prepare_tool_intermediate_steps(result_steps: list) -> List[Text]:
    """
    Constructs a list of ToolIntermediateStep objects from the given intermediate steps.
    Each intermediate step is expected to be a tuple containing a step (with attributes
    'tool' and 'tool_input') and its corresponding output.
    """
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


class Worker:
    @classmethod
    def create_model(cls, agent_description: Text, tools: List[Any], query: Union[Text, Dict[str, Any]]) -> AgentExecutor:
        llm = OllamaModel().raw_model()     
        sys_message = AGENT_SYSTEM_PROMPT
        if agent_description:
            agent_description = validate_input_variables(template=agent_description, input_variables=query)
            sys_message = f"AGENT DESCRIPTION:\n{agent_description}\n\nPROMPT:\n{AGENT_SYSTEM_PROMPT}"
        
        prompt = ChatPromptTemplate(
            [
                ("system", sys_message),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", AGENT_HUMAN_PROMPT),
            ]
        )

        agent = create_json_chat_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=max(4, 4 * len(tools)),
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )
        return agent_executor
    
    @classmethod
    def run_model(cls, worker: AgentExecutor, task_name: str):
        def run(state: StageExecute):
            team_iterations = state["team_iterations"] or 0
            result_steps = state.get("result_steps", [])
            agent_input = state["next_input"]

            try:
                agent_response = worker.invoke({
                    'input': agent_input,
                    'output_format': 'text'
                }, handle_parsing_errors=True)

                response_text = (
                    agent_response.get("output")
                    or agent_response.get("action_input")
                    or agent_response.get("input")
                    or str(agent_response)
                )

                # Determine if Final Answer was returned
                done = "final answer" in response_text.lower()

                tool_intermediate_steps = prepare_tool_intermediate_steps(
                    result_steps=agent_response.get("intermediate_steps", [])
                )

            except GraphRecursionError:
                response_text = "The agent reached the maximum number of iterations and could not solve the problem. Split the problem into multiple tasks."
                tool_intermediate_steps = []
                done = True  # Stop retrying

            result_steps.append(
                ResultStep(
                    agent=task_name,
                    input=agent_input,
                    output=response_text,
                    thought=response_text,
                    tool_steps=tool_intermediate_steps,
                )
            )

            next_node = "inspector" if done else task_name

            return {
                "result_steps": result_steps,
                "team_iterations": team_iterations + 1,
                "next": next_node,
                "next_input": response_text
            }

        return run
