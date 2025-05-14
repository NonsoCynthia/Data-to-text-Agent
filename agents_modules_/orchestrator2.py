import re
from typing import Dict, List, Text, Any, Optional
from langchain.agents import AgentExecutor
from ..utilities.utils import StageExecute, ResultStep
from ..agents.llm_model import OllamaModel
from ..agents.agent_prompts import ORCHESTRATOR_PROMPT
from agent_utils import  prepare_result_steps, render_intermediate_steps_xml, render_chat_history_xml, add_content_to_chat_history, prepare_chat_history_xml



class Orchestrator:
    @classmethod
    def create_model(cls) -> AgentExecutor:
        generator = OllamaModel()
        orchestrator = generator.model_(ORCHESTRATOR_PROMPT)
        return orchestrator


    @classmethod
    def run_model(cls, orchestrator: AgentExecutor, workers: List[Text]):
        def run(state: StageExecute):
            team_iterations = state["team_iterations"] or 0
            recursion_limit = state["recursion_limit"] or 50
            chat_history = state["chat_history"] or []
            chat_history = "\n".join(
                [f"{chats['role'].upper()}: {chats['content']}" for chats in chat_history]
            ).strip()
            inp = "INPUT\n--------------------\n" + "\n".join([chat_history, f"USER: {state['input']}"])

            result_steps = state["result_steps"] or []
            result_steps_str = prepare_result_steps(result_steps)
            if result_steps_str:
                inp = f"{'\n\n'.join(result_steps_str)}\n\n{inp}"

            if state["plan"] is not None:
                plan = state["plan"]
                plan_str = "\n".join(
                    f"Step {i+1}: Worker '{step['worker']}' - '{step['step']}'" for i, step in enumerate(plan)
                )
                inp = f"INITIAL PLAN:\n{plan_str}\n--------------------\n\n" + inp

            if (
                state["plan"] is not None
                and any(step["worker"] != "NOT_SOLVABLE" for step in state["plan"])
                or (state["plan"] is None and team_iterations < recursion_limit)
            ):
                agent_response = orchestrator.invoke({"input": inp}).content
                print(f"ORCHESTRATOR: {agent_response}")
                try:
                    thought, next, next_input = re.findall(
                        r"Thought:\s*(.*?)\s*Worker:\s*(.*?)\s*Worker Input:\s*(.*)", agent_response, re.DOTALL
                    )[0]
                except Exception:
                    thought, next, next_input = "", agent_response, agent_response
                response = next_input
            elif state["plan"] is None and team_iterations >= recursion_limit:
                thought, next, next_input = "", "FINISH", "Recursion limit reached."
                response = "incomplete"
            else:
                thought, next, next_input = "", "FINISH", "The assigned task cannot be solved by the team."
                response = "incomplete"

            next = next.replace(" tools", "").lower().strip()
            result_steps.append(
                ResultStep(
                    input=inp,
                    output=f"{next}(input='{next_input}')",
                    agent="orchestrator",
                    thought=thought,
                )
            )
            if next in workers:
                return {"next": next, 
                        "next_input": next_input,
                        "result_steps": result_steps,
                        "team_iterations": team_iterations + 1,
                        "recursion_limit": recursion_limit,}
            else:
                return {
                    "next": "inspector" if state["plan"] is not None else "FINISH",
                    "next_input": next_input,
                    "response": response,
                    "result_steps": result_steps,
                    "team_iterations": team_iterations + 1,
                    "recursion_limit": recursion_limit,
                }
        return run
