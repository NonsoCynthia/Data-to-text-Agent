import re
from enum import Enum
from uuid import uuid4
from typing import Annotated, Dict, List, Text, Any, Optional, Union, TypedDict, Tuple
from pydantic import BaseModel, Field


class ToolIntermediateStep(BaseModel):
    """Used to store intermediate steps in the tool's response."""
    tool: Text
    input: Union[List, Dict, Text]
    output: Union[List, Dict, Text]

class ResultStep(BaseModel):
    """Internally used to store result steps in the agent's response."""
    agent: Text
    input: Union[Text, Dict[str, Any]]
    output: Union[List, Dict, Text]
    thought: Optional[Union[List, Dict, Text]] = None
    
class StageExecute(TypedDict, total=False):  # set total=False to make all keys optional
    input: Union[Text, Dict[str, Any]]
    raw_input: Union[Text, Dict[str, Any]]
    plan: List[Dict[str, Any]]
    result_steps: List[ResultStep]
    chat_history: List[Dict[str, Any]]
    next: str
    next_input: str
    response: str
    agent_outcome: str
    inspector_feedback: str
    team_iterations: int
    recursion_limit: int
    current_step: int
    
# class StageExecute(TypedDict):
#     input: Union[Text, Dict[str, Any]]
#     plan: List[Dict[str, Any]]
#     result_steps: List[ResultStep] = []
#     chat_history: List[Dict[str, Any]] = []
#     next: str
#     next_input: str
#     response: str
#     agent_outcome: str
#     team_iterations: int = 0
#     recursion_limit: int = 100
#     current_step: int
    
# class Role(str, Enum):
#     USER = "user"
#     ASSISTANT = "assistant"
#     SYSTEM = "system"

#     def __str__(self):
#         return self._value_
    
# class Message(BaseModel):
#     id: Text = Field(default_factory=lambda: str(uuid4()))
#     role: Role
#     content: Text
#     result_steps: List[ResultStep] = []
    
# class AgentResponse(BaseModel):
#     """Agent Response to be returned to the user."""
#     input: Union[Text, Dict[str, Any]]
#     output: Any
#     feedback: Text = ""
#     result_steps: List[ResultStep] = []
#     plan: Optional[List[Dict[Text, Text]]] = None

# class Agent(BaseModel):
#     name: Text
#     role: Optional[Text] = None
#     instruction: Optional[Text] = None
#     description: Optional[Text] = None
#     tasks: Optional[List[Task]] = None
#     assets: List[Union[ModelTool, PipelineTool, UtilityTool, SQLTool]]
#     tools: Optional[List[ExternalTool]] = None
    
# class AgentResponseStatus(str, Enum):
#     SUCCESS = "SUCCESS"
#     FAILED = "FAILED"

# class AgentExecuteInput(BaseModel):
#     agent: Union[Agent, Text] 
#     query: Union[Text, Dict[str, Text]] #User input
#     chat_history: Optional[List[Message]] = None


