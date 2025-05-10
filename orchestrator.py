from typing import Dict, List, Text, Any, Optional
from langchain.agents import AgentExecutor
from utils import StageExecute, ResultStep
from llm_model import OllamaModel
from agent_prompts import ORCHESTRATOR_PROMPT
import re


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


class Orchestrator:
    @classmethod
    def create_model(cls) -> AgentExecutor:
        generator = OllamaModel()
        return generator.model_(ORCHESTRATOR_PROMPT)

    @classmethod
    def run_model(cls, orchestrator: AgentExecutor, workers: List[Text]):
        def run(state: StageExecute):
            result_steps = state.get("result_steps", [])
            team_iterations = state.get("team_iterations", 0)
            recursion_limit = state.get("recursion_limit", 50)
            chat_history = state.get("chat_history", [])
            plan = state.get("plan", [])
            current_step = state.get("current_step", 0)

            chat_str = "\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history).strip()
            inp = "INPUT\n--------------------\n" + "\n".join([chat_str, f"USER: {state['input']}"])

            result_steps_str = prepare_result_steps(result_steps)
            if result_steps_str:
                joined = "\n\n".join(result_steps_str)
                inp = f"{joined}\n\n{inp}"

            if plan:
                plan_str = "\n".join(
                    f"Step {i+1}: Worker '{step['worker']}' - '{step['step']}'"
                    for i, step in enumerate(plan)
                )
                inp = f"INITIAL PLAN:\n{plan_str}\n--------------------\n\n{inp}"

            if plan and current_step < len(plan):
                current_task = plan[current_step]
                next_worker = current_task["worker"]
                next_input = current_task["step"]
                thought = f"Proceeding to step {current_step + 1}: {next_worker} - {next_input}"
            elif current_step >= len(plan):
                next_worker = "FINISH"
                next_input = "All steps in the plan have been completed."
                thought = "All tasks complete."
            elif not plan and team_iterations >= recursion_limit:
                next_worker = "FINISH"
                next_input = "I couldn't provide an answer because the maximum number of iterations was reached."
                thought = "Recursion limit reached."
            else:
                next_worker = "FINISH"
                next_input = "The assigned task can not be solved by the team."
                thought = "No solvable plan found."

            result_steps.append(
                ResultStep(
                    input=inp,
                    output=f"{next_worker}(input='{next_input}')",
                    agent="orchestrator",
                    thought=thought,
                )
            )

            if next_worker in workers:
                return {
                    "next": next_worker,
                    "next_input": next_input,
                    "plan": plan,
                    "current_step": current_step,
                    "result_steps": result_steps,
                    "team_iterations": team_iterations + 1,
                    "recursion_limit": recursion_limit,
                }
            else:
                return {
                    "next": "inspector" if plan else "FINISH",
                    "next_input": next_input,
                    "response": next_input,
                    "plan": plan,
                    "current_step": current_step,
                    "result_steps": result_steps,
                    "team_iterations": team_iterations + 1,
                    "recursion_limit": recursion_limit,
                }

        return run
