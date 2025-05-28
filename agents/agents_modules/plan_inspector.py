import re
import json
from langchain.agents import AgentExecutor
from agents.utilities.utils import StageExecute, ResultStep
from agents.llm_model import UnifiedModel, model_name
from agents.agent_prompts import INSPECTOR_PROMPT, INSPECTOR_INPUT
from agents.utilities.agent_utils import prepare_result_steps, evaluate_with_comet_referenceless


class Plan_Inspector:
    @classmethod
    def create_model(cls, provider: str = "ollama") -> AgentExecutor:
        params = model_name.get(provider.lower())
        generator = UnifiedModel(
                            provider=provider,
                            model_name=params['model'],
                            temperature=params['temperature'],
                        )
        return generator.model_(INSPECTOR_PROMPT)

    @classmethod
    def run_model(cls, inspector: AgentExecutor):
        def run(state: StageExecute):
            chat_history = state.get("chat_history", [])
            chat_str = "\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history).strip()
            agent_input = f"INPUT\n--------------------\n{chat_str}\nUSER: {state['input']}".strip()
            data_input = state.get("raw_input", "")

            recursion_limit = state.get("recursion_limit", 50)
            team_iterations = state.get("team_iterations", 0) + 1
            result_steps = state.get("result_steps", [])
            plan = state.get("plan", [])

            solvable_steps = [step for step in plan if step.get("worker") != "NOT_SOLVABLE"]

            if solvable_steps:
                plan_str = "\n".join(
                    f"Step {i+1}: Worker '{step['worker']}' - '{step['step']}'"
                    for i, step in enumerate(plan)
                )
                result_steps_str = prepare_result_steps(result_steps)
                result_steps_str = "\n\n".join(result_steps_str) if result_steps_str else "No result steps."

                metric_result= ""
                if result_steps_str  == "surface realization":
                    metric = evaluate_with_comet_referenceless(input_data=data_input, prediction=result_steps_str)
                    metric_result  = f"Metric Evaluation Result: {metric}"

                # Compose input for inspector agent
                inspector_input = INSPECTOR_INPUT.format(
                    input=agent_input,
                    result_steps=result_steps_str,
                    plan=f"\nPlanning_steps: {plan_str}",
                    metric_result= metric_result
                    ).strip()
                
                print(f"INSPECTOR INPUT: {inspector_input}")

                agent_response = inspector.invoke({"input": inspector_input}).content
                feedback = agent_response.split("FEEDBACK:")[-1].strip()
                state["inspector_feedback"] = agent_response.split("FEEDBACK:")[-1].strip()

                print(f"INSPECTOR: {feedback}")

                if feedback == "correct":
                    # advance plan
                    updated_plan = plan[1:] if plan else []
                    output = "done" if not updated_plan else None
                else:
                    updated_plan = plan
                    output = "incomplete" if team_iterations >= recursion_limit else None
            else:
                feedback = "The assigned task cannot be solved by the team."
                updated_plan = []
                output = "done"

            result_steps.append(
                ResultStep(
                    input=agent_input,
                    output=feedback,
                    agent="inspector",
                    thought="",
                )
            )

            return {
                "response": output,
                "plan": updated_plan,
                "result_steps": result_steps,
                "team_iterations": team_iterations,
                "recursion_limit": recursion_limit,
                "next": "aggregator" if output == "done" else "orchestrator",
                "next_input": state.get("input", "")
            }

        return run
