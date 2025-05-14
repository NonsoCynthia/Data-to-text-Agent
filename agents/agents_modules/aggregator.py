from langchain.agents import AgentExecutor
from agents.utilities.utils import StageExecute, ResultStep
from agents.llm_model import OllamaModel
from agents.agent_prompts import AGGREGATOR_PROMPT, AGGREGATOR_INPUT
from agents.utilities.agent_utils import prepare_result_steps


# Implementing the abstract class
class ResponseAggregator:
    @classmethod
    def create_model(cls) -> AgentExecutor:
        generator = OllamaModel()
        aggregator = generator.model_(AGGREGATOR_PROMPT)
        return aggregator
    
    @classmethod
    def run_model(cls, aggregator: AgentExecutor):
        def run(state: StageExecute):
            chat_history = state["chat_history"] or []
            chat_history = "\n".join([f"{chats['role'].upper()}: {chats['content']}" for chats in chat_history]).strip()
            agent_input = "\n".join([chat_history, f"USER: {state['input']}"]).strip()
            
            result_steps = state["result_steps"] or []
            result_steps_str = prepare_result_steps(result_steps)    
            result_steps_str = ("\n\n".join(result_steps_str) if len(result_steps_str) > 0 else "No result steps.")
            
            plan = state["plan"]
            plan_str = "\n".join(f"Step {i+1}: Worker '{step['worker']}' - '{step['step']}'" for i, step in enumerate(plan) if "worker" in step and "step" in step)
            input_ = AGGREGATOR_INPUT.format(
                input=agent_input,
                result_steps=result_steps_str,
                plan=plan_str,
            )
            if state["response"] == "incomplete":
                agent_response = "I couldn't provide an answer because the maximum number of iterations was reached. Please try breaking the instruction into smaller questions by looking at the intermediate steps."
            else:
                agent_response = aggregator.invoke({'input': input_}).content
                print(f"AGGREGATOR: {agent_response}")

            result_steps.append(
                ResultStep(
                    input=agent_input,
                    output=agent_response,
                    agent="aggregator",
                )
            )
            return {
                "response": agent_response,
                "intermediate_steps": result_steps,
            }

        return run
