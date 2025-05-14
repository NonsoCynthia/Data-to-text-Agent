from langchain_core.prompts import ChatPromptTemplate
# from langchain_ollama.llms import OllamaLLM
from langchain_ollama import ChatOllama
from typing import Dict, List, Optional, Text

class OllamaModel:
    def __init__(self, model_name: str = "llama3.2", temperature: float = 0.3):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOllama(model=self.model_name, temperature=self.temperature)
        
    def model_(self, agent_prompts: Optional[Text]) -> Dict:
        human_prompt = "{input}"
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", agent_prompts),
                ("human", human_prompt),
            ]
        )
        agent_prompt = prompt | self.llm
        return agent_prompt
    
    def raw_model(self):
        return self.llm
    
    