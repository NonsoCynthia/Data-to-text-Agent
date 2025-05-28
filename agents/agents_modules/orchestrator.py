import re
from typing import Dict, List, Text, Any, Optional
from langchain.agents import AgentExecutor
from agents.utilities.utils import StageExecute, ResultStep
from agents.llm_model import UnifiedModel, model_name
from agents.agent_prompts import ORCHESTRATOR_PROMPT, ORCHESTRATOR_INPUT
from agents.utilities.agent_utils import get_inspector_validated_workers
from agents.utilities.agent_utils import (prepare_result_steps, render_chat_history_xml)

class Orchestrator:
    @classmethod
    def create_model(cls, provider: str = "ollama") -> AgentExecutor:
        params = model_name.get(provider.lower())
        generator = UnifiedModel(
            provider=provider,
            model_name=params["model"],
            temperature=params["temperature"],
        )
        return generator.model_(ORCHESTRATOR_PROMPT)

    @classmethod
    def run_model(cls, orchestrator: AgentExecutor, workers: List[Text], recursion_depth: Optional[int] = None):
        def run(state: StageExecute):
            team_iterations = state.get("team_iterations", 0)
            recursion_limit = recursion_depth or state.get("recursion_limit", 50)
            result_steps = state.get("result_steps", [])
            result_steps_str = "\n\n".join(prepare_result_steps(result_steps))

            chat_history = state.get("chat_history", [])
            chat_history = "\n".join([f"{turn['role'].upper()}: {turn['content']}" for turn in chat_history])
            user_request = "\n".join([chat_history, f"USER: {state.get('input') or ''}"])

            inspector_feedback = ""
            if state.get("inspector_feedback"):
                inspector_feedback = "\n".join([user_request, f"FEEDBACK: {state['inspector_feedback']}"])

            inp = ORCHESTRATOR_INPUT.format(
                input=user_request,
                result_steps=result_steps_str,
                feedback=f"INSPECTOR FEEDBACK: {inspector_feedback}"
            )

            response = "incomplete"
            if team_iterations < recursion_limit:
                agent_response = orchestrator.invoke({"input": inp}).content.strip()

                print(f"ORCHESTRATOR OUTPUT:\n{agent_response}\n")

                try:
                    thought, next_worker, next_input = re.findall(
                        r"Thought:\s*(.*?)\s*Worker:\s*(.*?)\s*Worker Input:\s*(.*)", agent_response, re.DOTALL
                    )[0]
                except Exception:
                    thought, next_worker, next_input = "Could not extract structure", "FINISH", agent_response

                response = next_input
            else:
                thought = "Max recursion limit reached."
                next_worker = "FINISH"
                next_input = "Exceeded maximum iterations."

            result_steps.append(
                ResultStep(
                    input=inp,
                    output=f"{next_worker}(input='{next_input}')",
                    agent="orchestrator",
                    thought=thought,
                )
            )

            if team_iterations == (recursion_limit - 1):
                return {
                    "next": "aggregator",
                    "next_input": next_input,
                    "response": response,
                    "result_steps": result_steps,
                    "team_iterations": team_iterations + 1,
                    "recursion_limit": recursion_limit,
                }


            if next_worker in workers:
                return {
                    "next": next_worker,
                    "next_input": next_input,
                    "response": response,
                    "result_steps": result_steps,
                    "team_iterations": team_iterations + 1,
                    "recursion_limit": recursion_limit,
                }
            elif next_worker == "FINISH":
                return {
                    "next": "aggregator",
                    "next_input": next_input,
                    "response": response,
                    "result_steps": result_steps,
                    "team_iterations": team_iterations + 1,
                    "recursion_limit": recursion_limit,
                }

        return run
