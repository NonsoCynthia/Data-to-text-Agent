import re
import json
from langchain.agents import AgentExecutor
from agents.utilities.utils import StageExecute, ResultStep
from agents.llm_model import UnifiedModel, model_name
from agents.agent_prompts import INSPECTOR_PROMPT, INSPECTOR_INPUT
from agents.utilities.agent_utils import prepare_result_steps


class Inspector:
    @classmethod
    def create_model(cls, provider: str = "ollama") -> AgentExecutor:
        params = model_name.get(provider.lower())
        generator = UnifiedModel(
                            provider=provider,
                            model_name=params['model'],
                            temperature=params['temperature'],
                        )
        inspector = generator.model_(INSPECTOR_PROMPT)
        return inspector

    @classmethod
    def run_model(cls, inspector: AgentExecutor):
        def run(state: StageExecute):
            chat_history = state["chat_history"] or []
            chat_history = "\n".join([f"{chats['role'].upper()}: {chats['content']}" for chats in chat_history]).strip()
            agent_input = "\n".join(["INPUT\n--------------------", chat_history, f"USER: {state['input']}"]).strip()
            
            recursion_limit = state["recursion_limit"] or 50
            team_iterations = state["team_iterations"] or 0
            team_iterations += 1

            result_steps = state["result_steps"] or []
            plan = state["plan"]
            solvable_steps = [step for step in plan if "worker" in step and step["worker"] != "NOT_SOLVABLE"]

            if len(solvable_steps) > 0:
                plan_str = "\n".join(
                    f"Step {i+1}: Worker '{step['worker']}' - '{step['step']}'"
                    for i, step in enumerate(plan)
                    if "worker" in step and "step" in step
                )
                result_steps_str = prepare_result_steps(result_steps)
                result_steps_str = "\n\n".join(result_steps_str) if result_steps_str else "No result steps."

                inspector_input = INSPECTOR_INPUT.format(
                    input=agent_input, result_steps=result_steps_str, plan=plan_str
                )
                agent_response = inspector.invoke({"input": inspector_input})
                feedback = agent_response.lower().split('feedback:')[-1].strip()

                thought = feedback

                # If the feedback is CORRECT, remove the completed step from the plan
                if feedback.upper() == "CORRECT":
                    updated_plan = plan[1:]  # Move to next step
                    output = "done" if not updated_plan else None
                else:
                    updated_plan = plan  # Keep plan unchanged
                    output = None if team_iterations < recursion_limit else "incomplete"

            else:
                thought = "The assigned task cannot be solved by the team."
                feedback = "done"
                updated_plan = []
                output = "done"

            result_steps.append(
                ResultStep(
                    input=agent_input,
                    output=feedback,
                    agent="inspector",
                    thought=thought,
                )
            )

            return {
                "response": output,
                "plan": updated_plan,
                "result_steps": result_steps,
                "team_iterations": team_iterations,
                "recursion_limit": recursion_limit,
            }

        return run
