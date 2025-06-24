import re
from langchain.agents import AgentExecutor
from agents.utilities.utils import ExecutionState, AgentStepOutput
from agents.llm_model import UnifiedModel, model_name
from agents.agent_prompts import (
    GUARDRAIL_PROMPT, 
    GUARDRAIL_INPUT,
    GUARDRAIL_PROMPT_CONTENT_SELECTION,
    GUARDRAIL_PROMPT_CONTENT_ORDERING,
    GUARDRAIL_PROMPT_TEXT_STRUCTURING,
    GUARDRAIL_PROMPT_SURFACE_REALIZATION,
    GUARDRAIL_PROMPT_FLUENCY_GRAMMAR,
    GUARDRAIL_PROMPT_FAITHFUL_ADEQUACY,
    GUARDRAIL_PROMPT_COHERENT_NATURAL,
    CONTENT_SELECTION_PROMPT,
    CONTENT_ORDERING_PROMPT, 
    TEXT_STRUCTURING_PROMPT, 
    SURFACE_REALIZATION_PROMPT,
)
# from agents.utilities.agent_utils import score_comet_quality

GUARDRAIL_TASKS = {
    "content selection": CONTENT_SELECTION_PROMPT,
    "content ordering": CONTENT_ORDERING_PROMPT,
    "text structuring": TEXT_STRUCTURING_PROMPT,
    "surface realization": SURFACE_REALIZATION_PROMPT,
}

class TaskGuardrail:
    provider = "openai"  # default

    @classmethod
    def init(cls, provider: str = "ollama") -> AgentExecutor:
        cls.provider = provider
        conf = model_name.get(provider.lower())
        return UnifiedModel(provider=provider, **conf).model_(GUARDRAIL_PROMPT)

    @classmethod
    def evaluate(cls, agent: AgentExecutor):
        def run(state: ExecutionState):
            history = state.get("history_of_steps", [])
            idx = state.get("iteration_count", 0)
            max_iter = state.get("max_iteration", 50)
            user_input = state.get("user_prompt", "")

            orch = next((s for s in reversed(history) if s.agent_name == "orchestrator"), None)
            worker = next((s for s in reversed(history) if s.agent_name not in ["orchestrator", "guardrail", "planner"]), None)

            task, task_input, output, rationale = "", "", "", ""
            if orch:
                rationale = orch.rationale
                match = re.search(r"^(.*?)\(input=['\"](.*?)['\"]\)$", orch.agent_output.strip(), re.DOTALL)
                if match:
                    task, task_input = match.groups()
            if worker:
                output = worker.agent_output

            prompt = GUARDRAIL_INPUT.format(
                input=f"""Worker: {task}\n
                Worker Description: {GUARDRAIL_TASKS.get(task.lower(), '').strip()}\n
                Orchestrator Thought: {rationale}\n
                Worker Input: {task_input}\n
                Worker Output: {output}""",
            )

            final_verdict = ""

            if task == "surface realization":
                conf = model_name.get(cls.provider)
                fluency_guard = UnifiedModel(cls.provider, **conf).model_(GUARDRAIL_PROMPT_FLUENCY_GRAMMAR)
                faithful_guard = UnifiedModel(cls.provider, **conf).model_(GUARDRAIL_PROMPT_FAITHFUL_ADEQUACY)
                coherence_guard = UnifiedModel(cls.provider, **conf).model_(GUARDRAIL_PROMPT_COHERENT_NATURAL)

                fluency_result = fluency_guard.invoke({"input": prompt}).content.strip().split("FEEDBACK:")[-1].strip()
                faith_result = faithful_guard.invoke({"input": prompt}).content.strip().split("FEEDBACK:")[-1].strip()
                coherence_result = coherence_guard.invoke({"input": prompt}).content.strip().split("FEEDBACK:")[-1].strip()

                # Construct structured review message
                review_message = (
                    "=== GUARDRAIL REVIEW (surface realization) ===\n"
                    f"[Fluency & Grammar]: {fluency_result}\n"
                    f"[Faithfulness & Adequacy]: {faith_result}\n"
                    f"[Coherence & Naturalness]: {coherence_result}\n"
                    f"OVERALL: {'CORRECT' if all(r.upper() == 'CORRECT' for r in [fluency_result, faith_result, coherence_result]) else f'Rerun {task} with feedback'}"
                )

                final_verdict = review_message


            # elif task == "content selection":
            #     conf = model_name.get(cls.provider)
            #     ordering_guard = UnifiedModel(cls.provider, **conf).model_(GUARDRAIL_PROMPT_CONTENT_SELECTION)
            #     result = ordering_guard.invoke({"input": prompt}).content.strip()
            #     final_verdict = result.split("FEEDBACK:")[-1].strip()
            
            elif task == "content ordering":
                conf = model_name.get(cls.provider)
                ordering_guard = UnifiedModel(cls.provider, **conf).model_(GUARDRAIL_PROMPT_CONTENT_ORDERING)
                result = ordering_guard.invoke({"input": prompt}).content.strip()
                final_verdict = result.split("FEEDBACK:")[-1].strip()

            elif task == "text structuring":
                conf = model_name.get(cls.provider)
                structuring_guard = UnifiedModel(cls.provider, **conf).model_(GUARDRAIL_PROMPT_TEXT_STRUCTURING)
                result = structuring_guard.invoke({"input": prompt}).content.strip()
                final_verdict = result.split("FEEDBACK:")[-1].strip()

            else:
                response = agent.invoke({"input": prompt}).content.strip()
                final_verdict = response.split("FEEDBACK:")[-1].strip()
            
            # print(final_verdict)

            history.append(AgentStepOutput(
                agent_name="guardrail",
                agent_input=prompt,
                agent_output=final_verdict,
                rationale="Evaluation of worker output."
            ))

            # done = final_verdict.upper() == "CORRECT" and task == "surface realization"
            done = final_verdict.strip().upper() == "CORRECT"
            return {"next_agent": "finalizer" if done else "orchestrator",
                    "response": "done" if done else None,
                    "history_of_steps": history,
                    "iteration_count": idx + 1,
                    "max_iteration": max_iter,
                    "next_agent_payload": user_input,
                    "review": final_verdict,
                }

        return run