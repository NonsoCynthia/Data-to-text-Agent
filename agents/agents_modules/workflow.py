from typing import List, Dict, Union, Any, Literal
from langgraph.graph import START, END, StateGraph
from agents.utilities.utils import ExecutionState
from agents.agents_modules.orchestrator import TaskOrchestrator
from agents.agents_modules.worker import TaskWorker
from agents.agents_modules.guardrail import TaskGuardrail
from agents.agents_modules.finalizer import TaskFinalizer
from agents.agent_prompts import (CONTENT_SELECTION_PROMPT, CONTENT_ORDERING_PROMPT, 
                                  TEXT_STRUCTURING_PROMPT, SURFACE_REALIZATION_PROMPT)


WORKER_ROLES = {
    # "content selection": CONTENT_SELECTION_PROMPT,
    "content ordering": CONTENT_ORDERING_PROMPT,
    "text structuring": TEXT_STRUCTURING_PROMPT,
    "surface realization": SURFACE_REALIZATION_PROMPT,
}

def add_workers(worker_prompts: Dict[str, str], graph: StateGraph, tools: List[Any], query: Union[str, Dict[str, Any]], provider: str) -> List[str]:
    added = []
    for name, prompt in worker_prompts.items():
        model = TaskWorker.init(description=prompt, tools=tools, context=query, provider=provider)
        graph.add_node(name, TaskWorker.execute(model, role=name))
        added.append(name)
    return added

def guardrail_routing(state: ExecutionState) -> Literal["orchestrator", "finalizer"]:
    expected = {
                # "content selection", 
                "content ordering", 
                "text structuring", 
                "surface realization"}
    done = {step.agent_name.lower() for step in state.get("history_of_steps", []) if step.agent_name.lower() in expected}
    return "finalizer" if expected.issubset(done) and "correct" in state.get("review", "").lower() else "orchestrator"

def build_agent_workflow(provider: str = "ollama") -> StateGraph:
    flow = StateGraph(ExecutionState)
    tools, query = [], ""
    workers = list(WORKER_ROLES.keys())

    flow.add_edge(START, "orchestrator")
    flow.set_entry_point("orchestrator")

    # Orchestrator
    flow.add_node("orchestrator", TaskOrchestrator.execute(TaskOrchestrator.init(provider)))

    # Workers
    add_workers(WORKER_ROLES, flow, tools, query, provider)

    # guardrail & Finalizer
    flow.add_node("guardrail", TaskGuardrail.evaluate(TaskGuardrail.init(provider)))
    flow.add_node("finalizer", TaskFinalizer.compile(TaskFinalizer.init(provider)))

    # Routing
    routes = {name: name for name in workers}
    routes.update({"finish": "finalizer"})
    flow.add_conditional_edges("orchestrator", lambda state: state["next_agent"], routes)
    for w in workers:
        flow.add_edge(w, "guardrail")
    flow.add_conditional_edges("guardrail", guardrail_routing)
    flow.add_edge("finalizer", END)

    return flow.compile()

# display(Image(process_flow.get_graph(xray=True).draw_mermaid_png()))
