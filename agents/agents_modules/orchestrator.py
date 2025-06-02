import re
from typing import Dict, List, Text, Any, Union, Optional
from agents.utilities.utils import ExecutionState, AgentStepOutput
from agents.llm_model import UnifiedModel, model_name
from agents.agent_prompts import ORCHESTRATOR_PROMPT, ORCHESTRATOR_INPUT
from agents.utilities.agent_utils import summarize_agent_steps

class TaskOrchestrator:
    @classmethod
    def init(cls, provider: str = "ollama"):
        conf = model_name.get(provider.lower())
        return UnifiedModel(provider=provider, **conf).model_(ORCHESTRATOR_PROMPT)

    @classmethod
    def execute(cls, executor: Any, roles: List[str], depth: Optional[int] = None):
        def run(state: ExecutionState):
            idx = state.get("iteration_count", 0)
            limit = depth or state.get("max_iteration", 50)
            history = state.get("history_of_steps", [])

            trace = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in state.get("dialogue_trace", [])])
            prompt = f"{trace}\nUSER: {state.get('user_prompt', '')}"
            feedback = f"\n{prompt}\nFEEDBACK: {state['review']}" if state.get("review") else ""
            summary = "\n\n".join(summarize_agent_steps(history))

            payload = ORCHESTRATOR_INPUT.format(
                input=prompt,
                result_steps=summary,
                feedback=f"INSPECTOR FEEDBACK: {feedback}"
            )

            if idx >= limit:
                return {
                    "next_agent": "aggregator",
                    "next_agent_payload": "Limit reached.",
                    "final_response": "stopped",
                    "history_of_steps": history,
                    "iteration_count": idx + 1,
                    "max_iteration": limit
                }

            output = executor.invoke({"input": payload}).content.strip()
            try:
                rationale, role, role_input = re.findall(r"Thought:\s*(.*?)\s*Worker:\s*(.*?)\s*Worker Input:\s*(.*)", output, re.DOTALL)[0]
            except Exception:
                rationale, role, role_input = "parse error", "FINISH", output

            history.append(AgentStepOutput(
                agent_name="orchestrator",
                agent_input=payload,
                agent_output=f"{role}(input='{role_input}')",
                rationale=rationale
            ))

            return {
                "next_agent": "aggregator" if role == "FINISH" else role,
                "next_agent_payload": role_input,
                "final_response": role_input,
                "history_of_steps": history,
                "iteration_count": idx + 1,
                "max_iteration": limit
            }
        return run
