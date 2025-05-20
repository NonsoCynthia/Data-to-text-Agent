import os
import json
from typing import List, Literal, Optional, Text, Union, Dict, Any
from IPython.display import Image, display
from langgraph.graph import START, END, StateGraph
from langchain_core.runnables.graph_mermaid import MermaidDrawMethod
from agents.utilities.utils import StageExecute, Agent, AgentExecuteInput
from agents.agents_modules.planner import Planner
from agents.agents_modules.plan_orchestrator import Plan_Orchestrator
from agents.agents_modules.orchestrator import Orchestrator
from agents.agents_modules.worker import Worker
from agents.agents_modules.inspector import Inspector
from agents.agents_modules.plan_inspector import Plan_Inspector
from agents.agents_modules.aggregator import ResponseAggregator
from agents.agent_prompts import content_ordering, text_structuring, surface_realization


# Worker role prompt descriptions
WORKER_DESCRIPTIONS = {
    "content ordering": content_ordering,
    "text structuring": text_structuring,
    "surface realization": surface_realization,
}

# Only keys are needed now — value is ignored and injected from above
workers_dict = {
    "content ordering": "",
    "text structuring": "",
    "surface realization": "",
}

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

# def transition_after_inspection(state: StageExecute) -> Literal["orchestrator", "aggregator"]:
#     """Decides the next stage after inspection based on task correctness."""
#     if state.get("response") in ["done", "incomplete"]:
#         return "aggregator"
#     return "orchestrator"


def add_workers(
    workers: Dict[str, str],  # Keys only are needed
    workflow: StateGraph,
    tools: List[Any],
    query: Union[str, Dict[str, Any]],
    provider: str = "ollama"
) -> List[str]:
    """
    For each worker name, injects the predefined agent role description,
    adds the worker as a node in the workflow, and returns all worker names.
    """
    worker_names = []
    for name in workers.keys():
        # Pull description from the standard map
        description = WORKER_DESCRIPTIONS.get(name.lower(), "")

        executor = Worker.create_model(
            agent_description=description,
            tools=tools,
            query=query,
            provider=provider,
        )

        workflow.add_node(
            name,
            Worker.run_model(executor, task_name=name)
        )

        worker_names.append(name)

    return worker_names



def build_agent_workflow(add_plan: bool, workers_dict: Dict[str, str], provider: str = "ollama") -> StateGraph:
    # 1. Create the StateGraph
    agent_workflow = StateGraph(StageExecute)

    # 2. Worker setup
    worker_names = list(workers_dict.keys())
    tools = []
    initial_query = ""

    # 6. Planner integration (optional)
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

    # 3. Add orchestrator
    orchestrator_node = orchestrator.run_model(orchestrator.create_model(provider=provider), 
                                               workers=worker_names)
    
    agent_workflow.add_node("orchestrator", orchestrator_node)

    # 4. Add workers
    add_workers(workers_dict, agent_workflow, tools, initial_query, provider=provider)

    # 5. Add inspector and aggregator
    agent_workflow.add_node("inspector", inspector.run_model(inspector.create_model(provider=provider)))
    agent_workflow.add_node("aggregator", ResponseAggregator.run_model(ResponseAggregator.create_model(provider=provider)))

    # 7. Orchestrator conditional routing
    conditional_map = {name: name for name in worker_names}
    conditional_map["inspector"] = "inspector"
    conditional_map["FINISH"] = "aggregator"
    agent_workflow.add_conditional_edges("orchestrator", lambda state: state["next"], conditional_map)

    # 8. Workers report to inspector
    for name in worker_names:
        agent_workflow.add_edge(name, "inspector")

    # 9. Inspector conditional routing
    agent_workflow.add_conditional_edges("inspector", transition_after_inspection)

    # 10. Final step
    agent_workflow.add_edge("aggregator", END)

    return agent_workflow.compile()


# process_flow = build_agent_workflow(add_plan=False, workers_dict=workers_dict)
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