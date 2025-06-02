import re
from langchain.agents import AgentExecutor
from agents.utilities.utils import ExecutionState, AgentStepOutput
from agents.llm_model import UnifiedModel, model_name
from agents.agent_prompts import (
    INSPECTOR_PROMPT, INSPECTOR_INPUT,
    CONTENT_ORDERING_PROMPT, TEXT_STRUCTURING_PROMPT, SURFACE_REALIZATION_PROMPT,
)
from agents.utilities.agent_utils import score_comet_quality

INSPECTOR_TASKS = {
    "content ordering": CONTENT_ORDERING_PROMPT,
    "text structuring": TEXT_STRUCTURING_PROMPT,
    "surface realization": SURFACE_REALIZATION_PROMPT,
}

class TaskInspector:
    @classmethod
    def init(cls, provider: str = "ollama") -> AgentExecutor:
        conf = model_name.get(provider.lower())
        return UnifiedModel(provider=provider, **conf).model_(INSPECTOR_PROMPT)

    @classmethod
    def evaluate(cls, agent: AgentExecutor):
        def run(state: ExecutionState):
            steps = state.get("history_of_steps", [])
            idx = state.get("iteration_count", 0)
            max_iter = state.get("max_iteration", 50)
            user_input = state.get("user_prompt", "")
            raw_data = state.get("raw_data", "")

            orch = next((s for s in reversed(steps) if s.agent_name == "orchestrator"), None)
            worker = next((s for s in reversed(steps) if s.agent_name not in ["orchestrator", "inspector", "planner"]), None)

            task, task_input, output, rationale = "", "", "", ""
            if orch:
                rationale = orch.rationale
                match = re.search(r"^(.*?)\(input=['\"](.*?)['\"]\)$", orch.agent_output.strip(), re.DOTALL)
                if match:
                    task, task_input = match.groups()
            if worker:
                output = worker.agent_output

            metric = ""
            if task == "surface realization":
                metric = score_comet_quality(raw_data, output)
                print(metric)

            prompt = INSPECTOR_INPUT.format(
                input=f"Worker: {task}\nWorker Description: {INSPECTOR_TASKS.get(task.lower(), '').strip()}\nOrchestrator Thought: {rationale}\nWorker Input: {task_input}\nWorker Output: {output}",
                metric_result=metric,
                result_steps="",
                plan=""
            )

            response = agent.invoke({"input": prompt}).content.strip()
            verdict = response.split("FEEDBACK:")[-1].strip()
            print(f"INSPECTOR: {verdict}")

            steps.append(AgentStepOutput(
                agent_name="inspector",
                agent_input=prompt,
                agent_output=verdict,
                rationale="Evaluation of worker output."
            ))

            done = verdict.upper() == "CORRECT" and task == "surface realization"
            return {
                "response": "done" if done else None,
                "history_of_steps": steps,
                "iteration_count": idx + 1,
                "max_iteration": max_iter,
                "next_agent": "aggregator" if done else "orchestrator",
                "next_agent_payload": user_input,
                "review": verdict,
            }
        return run
