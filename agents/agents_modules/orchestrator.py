import re
from typing import Dict, List, Text, Any, Optional
from langchain.agents import AgentExecutor
from agents.utilities.utils import StageExecute, ResultStep
from agents.llm_model import UnifiedModel, model_name
from agents.agent_prompts import ORCHESTRATOR_PROMPT
from agents.utilities.agent_utils import prepare_result_steps, add_content_to_chat_history, prepare_chat_history_xml


class Orchestrator:
    @classmethod
    def create_model(cls, provider: str = "ollama") -> AgentExecutor:
        params = model_name.get(provider.lower())
        generator = UnifiedModel(
                            provider=provider,
                            model_name=params['model'],
                            temperature=params['temperature'],
                        )
        prompt = ORCHESTRATOR_PROMPT
        return generator.model_(prompt)
    
    @classmethod
    def run_model(cls, orchestrator: AgentExecutor, workers: List[Text], recursion_depth: Optional[int] = None):
        def run(state: StageExecute):
            team_iterations = state.get("team_iterations", 0)
            recursion_limit = recursion_depth or state.get("recursion_limit", 50)

            # Update chat history
            chat_history = state.get("chat_history", [])
            chat_history = add_content_to_chat_history(chat_history, state.get("input", ""), "user")
            chat_history_xml = prepare_chat_history_xml(chat_history)

            # Prepare past result steps
            result_steps = state.get("result_steps", [])
            result_steps_str = "\n\n".join(prepare_result_steps(result_steps))

            # Compose input to the orchestrator
            inp = f"INPUT\n--------------------\n{chat_history_xml}\n\n{result_steps_str}".strip()

            # Run LLM orchestrator
            if team_iterations < recursion_limit:
                agent_response = orchestrator.invoke({"input": inp})
                raw_output = agent_response.content.strip()
                print(f"ORCHESTRATOR OUTPUT:\n{raw_output}\n")

                # Extract proposed step from LLM
                match = re.search(r"Thought:\s*(.*?)\s*Worker:\s*(.*?)\s*Worker Input:\s*(.*)", raw_output, re.DOTALL)
                if match:
                    thought = match.group(1).strip()
                    proposed_worker = match.group(2).strip().lower()
                    proposed_input = match.group(3).strip()
                else:
                    thought = "Failed to parse LLM output."
                    proposed_worker = "finish"
                    proposed_input = "Agent did not follow expected format."
                    raw_output = thought

                response = proposed_input
            else:
                thought = "Recursion limit reached."
                proposed_worker = "finish"
                proposed_input = "Maximum recursion limit reached."
                response = "incomplete"

            # Enforce ordered pipeline
            ordered_pipeline = ["content ordering", "text structuring", "surface realization"]
            completed_workers = [step.agent.lower() for step in result_steps if step.agent in ordered_pipeline]
            remaining_workers = [w for w in ordered_pipeline if w not in completed_workers]

            # Validate LLM's chosen worker
            if remaining_workers:
                expected_worker = remaining_workers[0]
                if proposed_worker != expected_worker:
                    print(f"[WARNING] Overriding LLM-proposed worker '{proposed_worker}' with expected '{expected_worker}'")
                    next_worker = expected_worker
                    next_input = result_steps[-1].output if result_steps else state.get("input", "")
                    next_input = str(next_input) if not isinstance(next_input, str) else next_input
                else:
                    next_worker = proposed_worker
                    next_input = proposed_input
            else:
                next_worker = "finish"
                next_input = "All required stages completed."

            # Log step
            result_steps.append(
                ResultStep(
                    input=inp,
                    output=f"{next_worker}(input='{next_input}')",
                    agent="orchestrator",
                    thought=thought,
                )
            )

            # Decide next routing
            if next_worker in workers:
                return {
                    "next": next_worker,
                    "next_input": next_input,
                    "result_steps": result_steps,
                    "team_iterations": team_iterations + 1,
                    "recursion_limit": recursion_limit,
                }
            elif next_worker == "finish":
                return {
                    "next": "aggregator",
                    "next_input": next_input,
                    "response": response,
                    "result_steps": result_steps,
                    "team_iterations": team_iterations + 1,
                    "recursion_limit": recursion_limit,
                }
            else:
                return {
                    "next": "inspector",
                    "next_input": next_input,
                    "response": f"Unrecognized worker '{next_worker}', falling back to inspector.",
                    "result_steps": result_steps,
                    "team_iterations": team_iterations + 1,
                    "recursion_limit": recursion_limit,
                }

        return run


    # @classmethod
    # def run_model(cls, orchestrator: AgentExecutor, workers: List[Text], recursion_depth: Optional[int] = None):
    #     def run(state: StageExecute):
    #         team_iterations = state.get("team_iterations", 0)
    #         recursion_limit = recursion_depth or state.get("recursion_limit", 50)

    #         # Update chat history
    #         chat_history = state.get("chat_history", [])
    #         chat_history = add_content_to_chat_history(chat_history, state.get("input", ""), "user")
    #         chat_history_xml = prepare_chat_history_xml(chat_history)

    #         # Prepare past result steps
    #         result_steps = state.get("result_steps", [])
    #         result_steps_str = "\n\n".join(prepare_result_steps(result_steps))

    #         # Compose input to the orchestrator
    #         inp = f"INPUT\n--------------------\n{chat_history_xml}\n\n{result_steps_str}".strip()

    #         # Invoke LLM agent
    #         if team_iterations < recursion_limit:
    #             agent_response = orchestrator.invoke({"input": inp})
    #             raw_output = agent_response.content.strip()
    #             print(f"ORCHESTRATOR OUTPUT:\n{raw_output}\n")

    #             # Extract values
    #             match = re.search(r"Thought:\s*(.*?)\s*Worker:\s*(.*?)\s*Worker Input:\s*(.*)", raw_output, re.DOTALL)
    #             if match:
    #                 thought = match.group(1).strip()
    #                 next_worker = match.group(2).strip().lower()
    #                 next_input = match.group(3).strip()
    #             else:
    #                 # Fallback to entire response
    #                 thought = ""
    #                 next_worker = "FINISH"
    #                 next_input = "Agent did not follow expected format."

    #             response = next_input
    #         else:
    #             thought = ""
    #             next_worker = "FINISH"
    #             next_input = "Maximum recursion limit reached."
    #             response = "incomplete"

    #         # Normalize worker
    #         next_worker = next_worker.replace(" tools", "").lower()

    #         # Log result step
    #         result_steps.append(
    #             ResultStep(
    #                 input=inp,
    #                 output=f"{next_worker}(input='{next_input}')",
    #                 agent="orchestrator",
    #                 thought=thought,
    #             )
    #         )

    #         # Decide next routing
    #         if next_worker in workers:
    #             return {
    #                 "next": next_worker,
    #                 "next_input": next_input,
    #                 "result_steps": result_steps,
    #                 "team_iterations": team_iterations + 1,
    #                 "recursion_limit": recursion_limit,
    #             }
    #         elif next_worker == "finish":
    #             return {
    #                 "next": "aggregator",
    #                 "next_input": next_input,
    #                 "response": response,
    #                 "result_steps": result_steps,
    #                 "team_iterations": team_iterations + 1,
    #                 "recursion_limit": recursion_limit,
    #             }
    #         else:
    #             return {
    #                 "next": "inspector",
    #                 "next_input": next_input,
    #                 "response": f"Unrecognized worker '{next_worker}', falling back to inspector.",
    #                 "result_steps": result_steps,
    #                 "team_iterations": team_iterations + 1,
    #                 "recursion_limit": recursion_limit,
    #             }

    #     return run
