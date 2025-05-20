import re
from langchain.agents import AgentExecutor
from agents.utilities.utils import StageExecute, ResultStep
from agents.llm_model import UnifiedModel, model_name
from agents.agent_prompts import INSPECTOR_PROMPT
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
        return generator.model_(INSPECTOR_PROMPT)

    @classmethod
    def run_model(cls, inspector: AgentExecutor):
        def run(state: StageExecute):
            result_steps = state.get("result_steps", [])
            team_iterations = state.get("team_iterations", 0)
            recursion_limit = state.get("recursion_limit", 50)
            chat_input = state.get("input", "")

            # Prepare the result steps as readable context
            result_steps_str = "\n\n".join(prepare_result_steps(result_steps)) or "No results."
            last_worker_step = result_steps[-1] if result_steps else None

            # Compose input for inspector agent
            inspector_input = f"""
You are an inspector reviewing the output of a worker agent in a data-to-text task.

Task: Determine whether the most recent worker's output is CORRECT.

Criteria:
- The output must reflect all data fields correctly.
- No hallucination (i.e., extra facts not present in the input).
- No missing fields.
- The response must be coherent and fulfill the task assigned to the worker.

Respond ONLY with one of the following:
- CORRECT
- One sentence explanation of the issue.

Input: {chat_input}

Completed Steps:
{result_steps_str}
""".strip()

            # Get feedback
            agent_response = inspector.invoke({"input": inspector_input}).content.strip()
            print(f"INSPECTOR: {agent_response}")

            # Determine status
            is_correct = agent_response.strip().upper() == "CORRECT"
            output = "done" if is_correct else ("incomplete" if team_iterations < recursion_limit else "halt")
            next_node = "orchestrator"

            # Log step
            result_steps.append(
                ResultStep(
                    input=inspector_input,
                    output=agent_response,
                    agent="inspector",
                    thought="Evaluation of last worker result.",
                )
            )

            return {
                "response": output,
                "result_steps": result_steps,
                "team_iterations": team_iterations + 1,
                "recursion_limit": recursion_limit,
                "next": next_node,
                "next_input": chat_input,
            }

        return run
