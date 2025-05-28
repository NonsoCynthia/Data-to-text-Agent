import re
from langchain.agents import AgentExecutor
from agents.utilities.utils import StageExecute, ResultStep
from agents.llm_model import UnifiedModel, model_name
from agents.agent_prompts import INSPECTOR_PROMPT, INSPECTOR_INPUT
from agents.utilities.agent_utils import evaluate_with_comet_referenceless


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
            data_input = state.get("raw_input", "")

            # Get last orchestrator step
            last_orchestrator = next((step for step in reversed(result_steps) if step.agent == "orchestrator"), None)
            orchestrator_thought = last_orchestrator.thought if last_orchestrator else "N/A"
            proposed_worker, proposed_input = "N/A", "N/A"

            if last_orchestrator and isinstance(last_orchestrator.output, str):
                match = re.search(r"^(.*?)\(input=['\"](.*?)['\"]\)$", last_orchestrator.output.strip(), re.DOTALL)
                if match:
                    proposed_worker = match.group(1).strip()
                    proposed_input = match.group(2).strip()

            # Get most recent worker step (exclude orchestrator/inspector/planner)
            last_worker_step = next(
                (step for step in reversed(result_steps)
                 if step.agent not in ["orchestrator", "inspector", "planner"]),
                None
            )
            worker_output = last_worker_step.output if last_worker_step else "N/A"

            # Evaluate with metric if surface realization was last run
            metric_result= ""
            if last_worker_step and last_worker_step.agent == "surface realization":
                try:
                    metric = evaluate_with_comet_referenceless(input_data=data_input, prediction=worker_output)
                    metric_result = f"Metric Evaluation Result: {metric}"
                except Exception as e:
                    metric_result = f"Metric Evaluation Failed: {str(e)}"

            # Format inspector input
            inspector_input = INSPECTOR_INPUT.format(
                input=f"""Worker: {proposed_worker}\nOrchestrator Thought: {orchestrator_thought}\nWorker Input: {proposed_input}\nWorker Output: {worker_output}""".strip(),
                metric_result=metric_result,
                result_steps="",
                plan=""
            ).strip()

            # print("*" * 50, "\n", f"INSPECTOR INPUT: {inspector_input}", "\n", "*" * 50)
            print(metric_result)

            agent_response = inspector.invoke({"input": inspector_input}).content.strip()
            feedback = agent_response.split("FEEDBACK:")[-1].strip()
            state["inspector_feedback"] = agent_response.split("FEEDBACK:")[-1].strip()

            print(f"INSPECTOR: {feedback}")

            next_worker = "orchestrator"
            workers_ = ["content ordering", "text structuring", "surface realization"]
            if feedback.upper().strip() == "CORRECT":
                if proposed_worker == "surface realization":
                    next_worker = "FINISH"
                else:
                    next_worker = "orchestrator"
            else:
                next_worker = "orchestrator"

            if team_iterations < recursion_limit:
                output = None
            if team_iterations >= recursion_limit:
                output = "incomplete"
            else:
                output = "done"

            result_steps.append(
                ResultStep(
                    input=inspector_input,
                    output=feedback,
                    agent="inspector",
                    thought="",
                )
            )

            return {
                "response": output,
                "result_steps": result_steps,
                "team_iterations": team_iterations + 1,
                "recursion_limit": recursion_limit,
                "next": next_worker,
                "next_input": chat_input,
                "feedback": feedback,
            }

        return run

