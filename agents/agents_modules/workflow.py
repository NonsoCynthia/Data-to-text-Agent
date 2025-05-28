import os
from typing import List, Literal, Optional, Text, Union, Dict, Any
from langgraph.graph import START, END, StateGraph
from agents.utilities.utils import StageExecute
from agents.agents_modules.planner import Planner
from agents.agents_modules.plan_orchestrator import Plan_Orchestrator
from agents.agents_modules.orchestrator import Orchestrator
from agents.agents_modules.worker import Worker
from agents.agents_modules.inspector import Inspector
from agents.agents_modules.plan_inspector import Plan_Inspector
from agents.agents_modules.aggregator import ResponseAggregator
from agents.agent_prompts import CONTENT_ORDERING_PROMPT, TEXT_STRUCTURING_PROMPT, SURFACE_REALIZATION_PROMPT
from agents.utilities.agent_utils import get_inspector_validated_workers

WORKER_DESCRIPTIONS = {
    "content ordering": CONTENT_ORDERING_PROMPT,
    "text structuring": TEXT_STRUCTURING_PROMPT,
    "surface realization": SURFACE_REALIZATION_PROMPT,
}

workers_dict = {
    "content ordering": "",
    "text structuring": "",
    "surface realization": "",
}

def add_workers(workers: Dict[str, str], workflow: StateGraph, tools: List[Any], query: Union[str, Dict[str, Any]], provider: str = "ollama") -> List[str]:
    worker_names = []
    for name in workers:
        executor = Worker.create_model(agent_description=WORKER_DESCRIPTIONS.get(name.lower(), ""), tools=tools, query=query, provider=provider)
        workflow.add_node(name, Worker.run_model(executor, task_name=name))
        worker_names.append(name)
    return worker_names


# def transition_after_inspection(state: StageExecute) -> Literal["orchestrator", "aggregator", "END"]:
#     validated = get_inspector_validated_workers(state["result_steps"])
#     required = {"content ordering", "text structuring", "surface realization"}

#     if state.get("response") == "done":
#         return "aggregator"
#     if required.issubset(validated):
#         return "aggregator"
#     return "orchestrator"

def transition_after_inspection(state: StageExecute) -> Literal["orchestrator", "aggregator"]:
    """
    After inspector validates the worker output, always return to orchestrator
    unless all required worker stages are completed and correct.
    """
    required_workers = {"content ordering", "text structuring", "surface realization"}
    completed_workers = {
        step.agent.lower()
        for step in state.get("result_steps", [])
        if step.agent.lower() in required_workers
    }

    if required_workers.issubset(completed_workers):
        return "aggregator"
    
    # Otherwise, continue looping back to orchestrator
    return "orchestrator"


def build_agent_workflow(add_plan: bool, workers_dict: Dict[str, str], provider: str = "ollama") -> StateGraph:
    agent_workflow = StateGraph(StageExecute)
    worker_names = list(workers_dict.keys())
    tools = []
    initial_query = ""

    if add_plan:
        orchestrator = Plan_Orchestrator
        inspector = Plan_Inspector
        agent_workflow.add_node("planner", Planner.run_model(Planner.create_model(provider=provider)))
        agent_workflow.add_edge(START, "planner")
        agent_workflow.add_edge("planner", "orchestrator")
        agent_workflow.set_entry_point("planner")
    else:
        orchestrator = Orchestrator
        inspector = Inspector
        agent_workflow.add_edge(START, "orchestrator")
        agent_workflow.set_entry_point("orchestrator")

    orchestrator_node = orchestrator.run_model(orchestrator.create_model(provider=provider), workers=worker_names)
    agent_workflow.add_node("orchestrator", orchestrator_node)

    add_workers(workers_dict, agent_workflow, tools, initial_query, provider)

    agent_workflow.add_node("inspector", inspector.run_model(inspector.create_model(provider=provider)))
    agent_workflow.add_node("aggregator", ResponseAggregator.run_model(ResponseAggregator.create_model(provider=provider)))

    conditional_map = {name: name for name in worker_names}
    conditional_map["FINISH"] = "aggregator"
    conditional_map["inspector"] = "inspector"
    agent_workflow.add_conditional_edges("inspector", transition_after_inspection)

    agent_workflow.add_conditional_edges("orchestrator", lambda state: state['next'], conditional_map)

    for worker in worker_names:
        agent_workflow.add_edge(worker, "inspector")

    agent_workflow.add_edge("aggregator", END)

    return agent_workflow.compile()

# display(Image(process_flow.get_graph(xray=True).draw_mermaid_png()))


# # === Test it out ===
# data = """<page_title> Kieran Bew </page_title> <section_title> Television </section_title> <table> <cell> Hans Christian Andersen: My Life as a Fairytale <col_header> Show </col_header> </cell> <cell> Hallmark Entertainment <col_header> Notes </col_header> </cell> <cell> The Street <col_header> Year </col_header> </cell> <cell> Gary Parr <col_header> Show </col_header> </cell> <cell> Da Vinci's Demons <col_header> Show </col_header> </cell> <cell> Duke Alphonso of Calabria <col_header> Role </col_header> </cell> </table>"""

# ground_truth = "Kieran Bew appeared in Da Vinci's Demons as Duke of Calabria, as Gary Parr in The Street and in Hans Christian Andersen: My Life as a Fairytale for Hallmark Entertainment."

# query = f"""You are an agent designed to generate text from data for a data-to-text natural language generation. You can be provided data in the form of xml, table, meaning representations, graphs etc. 
# Your task is to generate the appropriate text given the data information without omitting any field or adding extra information in essence called hallucination.
# Here is the data generate text using table data:
# {data}"""

# initial_state = {
#     "input": query,
#     "plan": [],
#     "result_steps": [],
#     "response": "",
#     "chat_history": [],
#     "next": "",
#     "agent_outcome": "",
#     "current_step": 0,
#     "team_iterations": 10,
#     "recursion_limit": 60,
# }

# state = process_flow.invoke(initial_state, config={"recursion_limit": initial_state["recursion_limit"]})
# print(state)
# print("*"*50)
# print(state['response'])