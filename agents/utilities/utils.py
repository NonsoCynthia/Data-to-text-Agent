import re
from enum import Enum
from uuid import uuid4
from typing import Annotated, Dict, List, Text, Any, Optional, Union, TypedDict, Tuple
from pydantic import BaseModel, Field


class AgentStepOutput(BaseModel):
    """Internally used to store result steps in the agent's response."""
    agent_name: Text
    agent_input: Union[Text, Dict[str, Any]]
    agent_output: Union[List, Dict, Text]
    rationale: Optional[Union[List, Dict, Text]] = None

    
class ExecutionState(TypedDict, total=False):  # set total=False to make all keys optional
    """Holds evolving state across agent pipeline execution."""
    user_prompt: Union[Text, Dict[str, Any]]
    final_response: str
    raw_data: Union[Text, Dict[str, Any]]
    next_agent: str
    next_agent_payload: str
    review: str
    iteration_count: int
    max_iteration: int
    current_step: int
    history_of_steps: List[AgentStepOutput]
