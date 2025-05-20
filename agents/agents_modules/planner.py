import re
import json
from typing import Optional
from langchain.agents import AgentExecutor
from agents.utilities.utils import StageExecute, ResultStep
from agents.llm_model import UnifiedModel, model_name
from agents.agent_prompts import PLANNER_PROMPT


# Implementing the abstract class
class Planner:
    @classmethod
    def create_model(cls, provider: str = "ollama") -> AgentExecutor:
        params = model_name.get(provider.lower())
        generator = UnifiedModel(
                            provider=provider,
                            model_name=params['model'],
                            temperature=params['temperature'],
                        )
        planner = generator.model_(PLANNER_PROMPT)
        return planner
    
    @classmethod
    def run_model(cls, planner: AgentExecutor): #, recursion_depth: Optional[int] = None
        def run(state: StageExecute):
            chat_history = state["chat_history"] or []
            chat_history = "\n".join([f"{chats['role'].upper()}: {chats['content']}" for chats in chat_history]).strip()
            agent_input = "\n".join(["INPUT\n----- ----- ----- -----", chat_history, f"USER: {state['input']}"]).strip()
            # print(agent_input)
            agent_response = planner.invoke({'input': agent_input}).content
            print(f"PLANNER: {agent_response}")
            
            # regular expression to extract thought and response
            try:
                thought, plans = re.findall(
                    r"Thought:\s*(.*?)\s*Plan:\s```json*(.*?)\s```", agent_response, re.DOTALL
                )[0]
                plans = json.loads(plans)
            except Exception:
                thought, plans = agent_response, []
            
            result_steps = state["result_steps"] or []
            team_iterations = state["team_iterations"] or 0
            # set recursion limit
            recursion_limit = state["recursion_limit"] or 60
            # recursion_limit = recursion_depth
            # if recursion_limit is None:
            #     recursion_limit = max(4, 4 * len(plans))
            
            result_steps.append(
                ResultStep(
                    agent="planner",
                    input=agent_input,
                    output=plans,
                    thought=thought,
                )
            )
            return {
                "plan": plans,
                "result_steps": result_steps,
                "team_iterations": team_iterations + 1,
                "recursion_limit": recursion_limit,
            }

        return run

# # Creating model and executing the inner function
# planner = Planner.create_model()
# run_function = Planner.run_model(planner)

# data = """<page_title> 2009 NASCAR Camping World Truck Series </page_title> <section_title> Schedule </section_title> <table> <cell> August 1 <col_header> Date </col_header> </cell> <cell> Toyota Tundra 200 <col_header> Event </col_header> </cell> <cell> Nashville Superspeedway <col_header> Venue </col_header> </cell> </table>"""

# query = f"""You are an agent designed to generate text from data for a data-to-text natural language generation. You can be provided data in the form of xml, table, meaning representations, graphs etc. 
# Your task is to generate the appropriate text given the data information without omitting any field or adding extra information in essence called hallucination.
# Here is the data:
# {data}"""

# # Define an initial state
# initial_state = {
#     "input": query,
#     "plan": [],
#     "result_steps": [],
#     "response": "",
#     "chat_history":"",
#     "next": "",
#     "agent_outcome": ""
# }

# # Execute the run function with the initial state
# final_state = run_function(initial_state)

# # Output final state
# print(final_state)

# from utils import prepare_result_steps

# # print(StageExecute["result_steps"])

# result_steps_str = prepare_result_steps(final_state["result_steps"])
# print("\n".join(result_steps_str))
