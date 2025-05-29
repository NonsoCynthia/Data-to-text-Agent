import re
from langchain.agents import AgentExecutor
from agents.utilities.utils import StageExecute, ResultStep
from agents.llm_model import UnifiedModel, model_name
from agents.agent_prompts import (
    INSPECTOR_PROMPT,
    INSPECTOR_INPUT,
    CONTENT_ORDERING_PROMPT,
    TEXT_STRUCTURING_PROMPT,
    SURFACE_REALIZATION_PROMPT,
)
from agents.utilities.agent_utils import evaluate_with_comet_referenceless

WORKER_PROMPTS = {
    "content ordering": CONTENT_ORDERING_PROMPT,
    "text structuring": TEXT_STRUCTURING_PROMPT,
    "surface realization": SURFACE_REALIZATION_PROMPT,
}


class Inspector:
    @classmethod
    def create_model(cls, provider: str = "ollama") -> AgentExecutor:
        params = model_name.get(provider.lower())
        generator = UnifiedModel(
            provider=provider,
            model_name=params["model"],
            temperature=params["temperature"],
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

            # Get last orchestrator and worker steps
            last_orchestrator = next((s for s in reversed(result_steps) if s.agent == "orchestrator"), None)
            last_worker = next((s for s in reversed(result_steps) if s.agent not in ["orchestrator", "inspector", "planner"]), None)

            proposed_worker = proposed_input = worker_output = orchestrator_thought = "N/A"
            if last_orchestrator:
                orchestrator_thought = last_orchestrator.thought
                match = re.search(r"^(.*?)\(input=['\"](.*?)['\"]\)$", last_orchestrator.output.strip(), re.DOTALL)
                if match:
                    proposed_worker = match.group(1).strip()
                    proposed_input = match.group(2).strip()

            if last_worker:
                worker_output = last_worker.output

            # Run metric evaluation if surface realization
            metric_result = ""
            if proposed_worker == "surface realization":
                try:
                    metric = evaluate_with_comet_referenceless(data_input, worker_output)
                    metric_result = f"Metric Evaluation Result: {metric}"
                except Exception as e:
                    metric_result = f"Metric Evaluation Failed: {str(e)}"

            # Get prompt for worker description
            worker_prompt = WORKER_PROMPTS.get(proposed_worker.lower(), "").strip()

            # Format inspector input
            inspector_input = INSPECTOR_INPUT.format(
                input=(
                    f"Worker: {proposed_worker}\n"
                    f"Worker Description: {worker_prompt}\n"
                    f"Orchestrator Thought: {orchestrator_thought}\n"
                    f"Worker Input: {proposed_input}\n"
                    f"Worker Output: {worker_output}"
                ).strip(),
                metric_result=metric_result,
                result_steps="",
                plan="",
            )

            # print("*" * 50, "\n", f"INSPECTOR INPUT: {inspector_input}", "\n", "*" * 50)
            print(metric_result)

            # Run inspector
            agent_response = inspector.invoke({"input": inspector_input}).content.strip()
            feedback = agent_response.split("FEEDBACK:")[-1].strip()
            state["inspector_feedback"] = feedback

            print(f"INSPECTOR: {feedback}")

            # Determine next step
            is_correct = feedback.upper().strip() == "CORRECT"
            next_worker = "FINISH" if is_correct and proposed_worker == "surface realization" else "orchestrator"

            if team_iterations >= recursion_limit:
                output = "incomplete"
            elif is_correct and proposed_worker == "surface realization":
                output = "done"
            else:
                output = None

            # Log inspector result
            result_steps.append(
                ResultStep(
                    input=inspector_input,
                    output=feedback,
                    agent="inspector",
                    thought="Evaluation of last worker result.",
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
