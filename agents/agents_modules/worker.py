from typing import Dict, List, Text, Any, Union, Tuple
from langchain.agents import AgentExecutor, create_json_chat_agent
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langgraph.errors import GraphRecursionError
from agents.utilities.utils import StageExecute, ResultStep, ToolIntermediateStep
from agents.llm_model import UnifiedModel, model_name
from agents.agent_prompts import WORKER_PROMPT, AGENT_SYSTEM_PROMPT, AGENT_HUMAN_PROMPT
from agents.utilities.agent_utils import validate_input_variables, prepare_tool_result_steps



class Worker:
    @classmethod
    def create_model(cls, agent_description: Text, tools: List[Any], query: Union[Text, Dict[str, Any]], provider: str = "ollama") -> AgentExecutor:
        params = model_name.get(provider.lower())
        llm = UnifiedModel(
                            provider=provider,
                            model_name=params['model'],
                            temperature=params['temperature'],
                        ).raw_model()
        sys_message = AGENT_SYSTEM_PROMPT

        if agent_description:
            agent_description = validate_input_variables(template=agent_description, input_variables=query)
            sys_message = f"AGENT DESCRIPTION:\n{agent_description}\n\nPROMPT:\n{AGENT_SYSTEM_PROMPT}"

        prompt = ChatPromptTemplate.from_messages([
            ("system", sys_message),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", AGENT_HUMAN_PROMPT),
        ])
        prompt = prompt.partial(output_format="text")

        agent = create_json_chat_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=max(4, 4 * len(tools)),
            handle_parsing_errors=True,
            return_result_steps=True,
        )
        return agent_executor

    @classmethod
    def run_model(cls, worker: AgentExecutor, task_name: str):
        def run(state: StageExecute):
            team_iterations = state.get("team_iterations", 0)
            result_steps = state.get("result_steps", [])
            agent_input = state.get("next_input", "")

            try:
                result = worker.invoke({"input": agent_input})
                print(f"WORKER: {result}")
                if isinstance(result, dict):
                    response_text = result.get("output") or result.get("action_input") or str(result)
                    tool_result_steps = prepare_tool_result_steps(result.get("result_steps", []))
                elif hasattr(result, "content"):
                    response_text = result.content
                    tool_result_steps = []
                else:
                    response_text = str(result)
                    tool_result_steps = []

            except GraphRecursionError:
                response_text = "The agent reached the maximum number of iterations and could not solve the problem. Split the problem into multiple tasks."
                tool_result_steps = []

            result_steps.append(
                ResultStep(
                    agent=task_name,
                    input=agent_input,
                    output=response_text,
                    thought=response_text,
                    tool_steps=tool_result_steps,
                )
            )

            return {
                "result_steps": result_steps,
                "team_iterations": team_iterations + 1
            }

        return run
