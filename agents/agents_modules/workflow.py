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

# === Descriptions for Workers ===
WORKER_DESCRIPTIONS = {
    "content ordering": CONTENT_ORDERING_PROMPT,
    "text structuring": TEXT_STRUCTURING_PROMPT,
    "surface realization": SURFACE_REALIZATION_PROMPT,
}

# === List of Worker Roles ===
workers_dict = {
    "content ordering": "",
    "text structuring": "",
    "surface realization": "",
}


# === Add Worker Nodes ===
def add_workers(workers: Dict[str, str], workflow: StateGraph, tools: List[Any], query: Union[str, Dict[str, Any]], provider: str = "ollama") -> List[str]:
    worker_names = []
    for name in workers:
        executor = Worker.create_model(
            agent_description=WORKER_DESCRIPTIONS.get(name.lower(), ""),
            tools=tools,
            query=query,
            provider=provider
        )
        workflow.add_node(name, Worker.run_model(executor, task_name=name))
        worker_names.append(name)
    return worker_names


# === Route Logic After Inspection ===
def transition_after_inspection(state: StageExecute) -> Literal["orchestrator", "aggregator"]:
    required = {"content ordering", "text structuring", "surface realization"}
    completed = {step.agent.lower() for step in state.get("result_steps", []) if step.agent.lower() in required}
    feedback = state.get("feedback", "").lower()

    if required.issubset(completed) and "correct" in feedback:
        return "aggregator"
    return "orchestrator"

# === Workflow Definition ===
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

    # === Orchestrator ===
    orchestrator_node = orchestrator.run_model(
        orchestrator.create_model(provider=provider),
        workers=worker_names
    )
    agent_workflow.add_node("orchestrator", orchestrator_node)

    # === Workers ===
    add_workers(workers_dict, agent_workflow, tools, initial_query, provider=provider)

    # === Inspector & Aggregator ===
    agent_workflow.add_node("inspector", inspector.run_model(inspector.create_model(provider=provider)))
    agent_workflow.add_node("aggregator", ResponseAggregator.run_model(ResponseAggregator.create_model(provider=provider)))

    # === Routing: Orchestrator → Workers / Inspector / Aggregator
    orchestrator_map = {name: name for name in worker_names}
    orchestrator_map["inspector"] = "inspector"
    orchestrator_map["FINISH"] = "aggregator"
    agent_workflow.add_conditional_edges("orchestrator", lambda state: state["next"], orchestrator_map)

    # === Routing: Worker → Inspector
    for worker in worker_names:
        agent_workflow.add_edge(worker, "inspector")

    # === Routing: Inspector → Orchestrator or Aggregator
    agent_workflow.add_conditional_edges("inspector", transition_after_inspection)

    # === Routing: Aggregator → END
    agent_workflow.add_edge("aggregator", END)

    return agent_workflow.compile()

# display(Image(process_flow.get_graph(xray=True).draw_mermaid_png()))
