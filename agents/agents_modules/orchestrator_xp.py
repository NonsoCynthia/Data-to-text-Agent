__author__ = "lucaspavanelli"

import agentification.aixplain_prompts as aixplain_prompts
import re
import time

from agentification.aixplain_chat_model import AixplainChatModel, GroupInfo
from agentification.team_agent.node import NodeService
from agentification.utilities.models import IntermediateStep, PlanExecute
from agentification.utilities.utils import (
    prepare_intermediate_steps,
    prepare_chat_history_xml,
    RECURSION_LIMIT,
    add_content_to_chat_history,
)
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Optional, Text


class OrchestratorService(NodeService):
    @classmethod
    def create(
        cls,
        llm_id: Text,
        members: List[Text],
        tools: List[Text],
        api_key: Text,
        max_tokens: int = 2048,
        group_info: Optional[GroupInfo] = None,
        instructions: Optional[Text] = None,
    ) -> AgentExecutor:
        members = "\n".join(members)
        tools = "\n\n".join(tools)

        sys_prompt = aixplain_prompts.ORCHESTRATOR_PROMPT
        if instructions is not None:
            sys_prompt = "\n".join(["INSTRUCTIONS:", instructions, "\nPROMPT:", sys_prompt])

        human_prompt = """{input}"""

        prompt = ChatPromptTemplate.from_messages([("system", sys_prompt), ("human", human_prompt)]).partial(
            members=members, tools=tools
        )

        model = AixplainChatModel(model_id=llm_id, api_key=api_key, max_tokens=max_tokens, group_info=group_info)
        orchestrator = prompt | model
        return orchestrator

    @classmethod
    def create_run(
        cls,
        orchestrator: AgentExecutor,
        workers: List[Text],
        recursion_depth: Optional[int] = None,
        use_inspector: bool = False,
    ):  # TODO: inspector without mentalist
        # Define orchestrator
        def run(state: PlanExecute):
            start = time.time()
            # api calls
            api_calls = 0
            # recursion limit
            recursion_limit = recursion_depth
            if recursion_limit is None:
                recursion_limit = state.get("recursion_limit") or RECURSION_LIMIT
            # intermediate steps
            intermediate_steps = state.get("intermediate_steps") or []
            intermediate_steps_str = prepare_intermediate_steps(intermediate_steps)
            # chat history
            chat_history = state.get("chat_history") or []
            chat_history = add_content_to_chat_history(chat_history, state.get("input") or "", "user")
            chat_history_xml = prepare_chat_history_xml(chat_history)

            # Format the input with XML chat history
            inp = "INPUT\n--------------------\n"
            inp += chat_history_xml
            inp = f"""{inp}\n\n{intermediate_steps_str}""".strip()

            team_iterations = state.get("team_iterations") or 0

            if team_iterations < recursion_limit:
                agent_response = orchestrator.invoke({"input": inp})
                api_calls += 1
                # regular expression to extract thought and response
                try:
                    thought, next, next_input = re.findall(
                        r"Thought:\s*(.*?)\s*Worker:\s*(.*?)\s*Worker Input:\s*(.*)", agent_response.content, re.DOTALL
                    )[0]
                except Exception:
                    thought, next, next_input = "", agent_response.content, agent_response.content
                used_credits = agent_response.response_metadata["used_credits"]
                response = next_input
            else:
                thought = ""
                next = "FINISH"
                next_input = (
                    next_input
                ) = "I couldn't provide an answer because the maximum number of iterations was reached. Please try breaking the instruction into smaller questions by looking at the intermediate steps."
                response = "incomplete"
                used_credits = 0

            # remove potential extra tools word from 'next'
            # TODO make sure orchestrator prompt is handling this
            next = next.replace(" tools", "").strip()
            end = time.time()
            new_intermediate_steps = [
                IntermediateStep(
                    input=inp,
                    output=f"{next}(input='{next_input}')",
                    agent="orchestrator",
                    thought=thought,
                    runTime=round(end - start, 3),
                    usedCredits=used_credits,
                    apiCalls=api_calls,
                )
            ]
            if next in workers:
                return {
                    "next": f"{next}_agent",
                    "next_input": next_input,
                    "intermediate_steps": new_intermediate_steps,
                    "team_iterations": team_iterations + 1,
                    "recursion_limit": recursion_limit,
                }
            else:
                return {
                    "next": "FINISH",
                    "next_input": next_input,
                    "response": response,
                    "intermediate_steps": new_intermediate_steps,
                    "team_iterations": team_iterations + 1,
                    "recursion_limit": recursion_limit,
                }

        return run
