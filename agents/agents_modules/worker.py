from typing import Dict, List, Text, Any, Union, Optional
from langchain.agents import AgentExecutor, create_json_chat_agent
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langgraph.errors import GraphRecursionError
from agents.utilities.utils import ExecutionState, AgentStepOutput
from agents.llm_model import UnifiedModel, model_name
from agents.agent_prompts import WORKER_SYSTEM_PROMPT, WORKER_HUMAN_PROMPT
from agents.utilities.agent_utils import apply_variable_substitution


class TaskWorker:
    @classmethod
    def init(cls, description: Text, tools: List[Any], context: Union[Text, Dict[str, Any]], provider: str = "ollama") -> AgentExecutor:

        params = model_name.get(provider.lower())
        model = UnifiedModel(provider=provider, **params).raw_model()
        desc = apply_variable_substitution(description, context) if description else ""

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"AGENT DESCRIPTION:\n{desc}\n\nEXECUTION INSTRUCTION:\n{WORKER_SYSTEM_PROMPT}" if desc else WORKER_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", WORKER_HUMAN_PROMPT),
        ]).partial(output_format="text")
        
        return AgentExecutor(
            agent=create_json_chat_agent(model, tools, prompt),
            tools=tools,
            verbose=True,
            max_iterations=max(4, 4 * len(tools)),
            handle_parsing_errors=True,
            return_result_steps=True,
        )

    @classmethod
    def execute(cls, agent: AgentExecutor, role: str):
        def run(state: ExecutionState):
            idx = state.get("iteration_count", 0)
            inputs = state.get("next_agent_payload", "")
            history = state.get("history_of_steps", [])
            try:
                out = agent.invoke({"input": inputs})
                text = out.get("output") or out.get("action_input") or getattr(out, "content", str(out))
                tools = out.get("result_steps", []) if isinstance(out, dict) else []
            except GraphRecursionError:
                text, tools = "Too many iterations. Try splitting task.", []

            history.append(AgentStepOutput(
                agent_name=role,
                agent_input=inputs,
                agent_output=text,
                rationale=text,
                tool_steps=tools
            ))
            return {"next_agent": "guardrail",
                    "history_of_steps": history, 
                    "iteration_count": idx + 1
                    }
        return run
