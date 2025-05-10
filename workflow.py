from langgraph.graph import START, END, StateGraph
from utils import StageExecute, Agent, AgentExecuteInput
from langgraph.graph import START, END, StateGraph
from utils import StageExecute, Agent, AgentExecuteInput
from IPython.display import Image, display
from typing import List, Literal, Optional, Text, Union, Dict, Any
from planner import Planner
from orchestrator import Orchestrator
from worker import Worker
from inspector import Inspector
from aggregator import ResponseAggregator
from agent_prompts import content_ordering, text_structuring, surface_realization


workers = {
    "content ordering": content_ordering,
    "text structuring": text_structuring,
    "surface realization": surface_realization,
}

def transition_after_inspection(state: StageExecute) -> Literal["orchestrator", "aggregator"]:
    """Decides the next stage after inspection based on task correctness."""
    if state.get("response") in ["done", "incomplete"]:
        return "aggregator"
    return "orchestrator"

# ——— Helper to add your two workers into the graph ———
def add_workers(
    workers: Dict[str, str],
    workflow: StateGraph,
    tools: List[Any],
    query: Union[str, Dict[str, Any]],
) -> List[str]:
    """
    For each (name, description) in `workers`, creates a Worker agent,
    adds it as a node to `workflow`, and returns the list of names.
    """
    worker_names = []
    for name, description in workers.items():
        # 1) create the AgentExecutor for this worker
        executor = Worker.create_model(
            agent_description=description,
            tools=tools,
            query=query,
        )
        # 2) wrap it in your run_model and add to the graph
        workflow.add_node(
            name,
            Worker.run_model(executor, task_name=name)
        )
        worker_names.append(name)
    return worker_names

workers_list = [key for key, value in workers.items()]

# ——— Build your StateGraph ———
agent_workflow = StateGraph(StageExecute)

# 1) planner node
agent_workflow.add_node("planner", Planner.run_model(Planner.create_model()))

# 2) orchestrator node
agent_workflow.add_node(
    "orchestrator",
    Orchestrator.run_model(Orchestrator.create_model(), workers=workers_list)
)

# 3) worker nodes
#    here `tools` can be [] or any list of LangChain tools you have;
#    `initial_query` is whatever you're passing in as the user input.
tools = []
initial_query = StageExecute["input"]  # or whatever your entry payload is
worker_names = add_workers(workers, agent_workflow, tools, initial_query)
agent_workflow.add_node("inspector", Inspector.run_model(Inspector.create_model()))
agent_workflow.add_node("aggregator", ResponseAggregator.run_model(ResponseAggregator.create_model()))


# 4) wire up the edges
#  start → planner → orchestrator
agent_workflow.add_edge(START, "planner")
agent_workflow.add_edge("planner", "orchestrator")

#  orchestrator uses the "next" field in state to pick which worker to run
#  we map each worker name to itself, and FINISH to END
conditional_map = {name: name for name in worker_names}
conditional_map["inspector"] = "inspector"  # Add inspector explicitly

agent_workflow.add_conditional_edges(
    "orchestrator",
    lambda state: state["next"],
    conditional_map
)

#  after a worker runs, end the graph
for name in worker_names:
    agent_workflow.add_edge(name, "inspector")
    
agent_workflow.add_conditional_edges("inspector", transition_after_inspection)
agent_workflow.add_edge("aggregator", END)

#  set the entry point
agent_workflow.set_entry_point("planner")

#  compile into a runnable app
process_flow = agent_workflow.compile()

# display(Image(process_flow.get_graph(xray=True).draw_mermaid_png()))


# === Test it out ===
data = """<page_title> Kieran Bew </page_title> <section_title> Television </section_title> <table> <cell> Hans Christian Andersen: My Life as a Fairytale <col_header> Show </col_header> </cell> <cell> Hallmark Entertainment <col_header> Notes </col_header> </cell> <cell> The Street <col_header> Year </col_header> </cell> <cell> Gary Parr <col_header> Show </col_header> </cell> <cell> Da Vinci's Demons <col_header> Show </col_header> </cell> <cell> Duke Alphonso of Calabria <col_header> Role </col_header> </cell> </table>"""

ground_truth = "Kieran Bew appeared in Da Vinci's Demons as Duke of Calabria, as Gary Parr in The Street and in Hans Christian Andersen: My Life as a Fairytale for Hallmark Entertainment."

query = f"""You are an agent designed to generate text from data for a data-to-text natural language generation. You can be provided data in the form of xml, table, meaning representations, graphs etc. 
Your task is to generate the appropriate text given the data information without omitting any field or adding extra information in essence called hallucination.
Here is the data generate text using table data:
{data}"""

initial_state = {
    "input": query,
    "plan": [],
    "result_steps": [],
    "response": "",
    "chat_history": [],
    "next": "",
    "agent_outcome": "",
    "current_step": 0,
    "team_iterations": 10,
    "recursion_limit": 60,
}

state = process_flow.invoke(initial_state, config={"recursion_limit": initial_state["recursion_limit"]})
print(state)
print("*"*50)
print(state['response'])