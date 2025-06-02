from langchain.agents import AgentExecutor
from agents.utilities.utils import ExecutionState, AgentStepOutput
from agents.llm_model import UnifiedModel, model_name
from agents.agent_prompts import AGGREGATOR_PROMPT, AGGREGATOR_INPUT
from agents.utilities.agent_utils import summarize_agent_steps

class TaskAggregator:
    @classmethod
    def init(cls, provider: str = "ollama") -> AgentExecutor:
        cfg = model_name.get(provider.lower())
        return UnifiedModel(provider=provider, **cfg).model_(AGGREGATOR_PROMPT)

    @classmethod
    def compile(cls, executor: AgentExecutor):
        def run(state: ExecutionState):
            trace = state.get("dialogue_trace", [])
            user_msg = f"\n".join(f"{m['role'].upper()}: {m['content']}" for m in trace)
            prompt = f"{user_msg}\nUSER: {state.get('user_prompt', '')}"
            plan = state.get("execution_plan", [])
            plan_str = "\n".join(f"Step {i+1}: Worker '{p['worker']}' - '{p['step']}'" for i, p in enumerate(plan))
            steps = state.get("history_of_steps", [])
            step_log = "\n\n".join(summarize_agent_steps(steps)) or "No result steps."

            final_input = AGGREGATOR_INPUT.format(input=prompt, result_steps=step_log, plan=plan_str)

            if state.get("response") == "incomplete":
                reply = "I couldn't provide an answer because the maximum number of iterations was reached. Please try breaking the instruction into smaller questions by looking at the intermediate steps."
            else:
                reply = executor.invoke({"input": final_input}).content.replace("Final Answer:", "").strip()
                print(f"AGGREGATOR: {reply}")

            steps.append(AgentStepOutput(
                agent_name="aggregator",
                agent_input=prompt,
                agent_output=reply
            ))

            return {"final_response": reply, "history_of_steps": steps}
        return run
