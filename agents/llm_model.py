import os
from dotenv import load_dotenv, find_dotenv
from typing import Dict, Optional, Text, Union
from langchain_core.prompts import ChatPromptTemplate

_ = load_dotenv(find_dotenv())  # Load environment variables
# hf_token = os.getenv("HF_TOKEN")
# openai_key = os.getenv("OPENAI_KEY")

# === Base Interface ===
class ModelBase:
    def model_(self, agent_prompts: Optional[Text]) -> Dict:
        raise NotImplementedError

    def raw_model(self):
        raise NotImplementedError


# === Ollama Model ===
class OllamaModel(ModelBase):
    def __init__(self, model_name: str = "llama3.2", temperature: float = 0.3):
        from langchain_ollama import ChatOllama
        self.llm = ChatOllama(model=model_name, temperature=temperature)

    def model_(self, agent_prompts: Optional[Text]) -> Dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", agent_prompts), ("human", "{input}")
        ])
        return prompt | self.llm

    def raw_model(self):
        return self.llm


# === OpenAI Model ===
class OpenAIModel(ModelBase):
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0, api_key: Optional[str] = None):
        from langchain_openai import ChatOpenAI
        openai_key = os.getenv("OPENAI_KEY") or api_key
        self.llm = ChatOpenAI(model=model_name, temperature=temperature, api_key=openai_key)

    def model_(self, agent_prompts: Optional[Text]) -> Dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", agent_prompts), ("human", "{input}")
        ])
        return prompt | self.llm

    def raw_model(self):
        return self.llm
    

class AiXplainModel(ModelBase):
    def __init__(self, model_id: str = "640b517694bf816d35a59125", temperature: float = 0.3, api_key: Optional[str] = None):
        from aixplain.factories import ModelFactory
        os.environ["TEAM_API_KEY"] = os.getenv("TEAM_API_KEY") or api_key
        self.llm = ModelFactory.get(model_id)
        self.temperature = temperature  # store if you need to use it in prompts

    def model_(self, agent_prompts: Optional[Text]) -> Dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", agent_prompts), ("human", "{input}")
        ])
        return prompt | self.llm

    def raw_model(self):
        return self.llm

# === HuggingFace Model ===

class HFModel(ModelBase):
    def __init__(self, model_name: str = "HuggingFaceH4/zephyr-7b-beta", temperature: float = 0.3, api_key: Optional[str] = None):
        from langchain_huggingface import ChatHuggingFace
        hf_token = os.getenv("HF_TOKEN") or api_key
        self.llm = ChatHuggingFace(
            model=model_name,
            temperature=temperature,
            huggingfacehub_api_token=hf_token
        )

    def model_(self, agent_prompts: Optional[Text]) -> Dict:
        prompt = ChatPromptTemplate.from_messages([
            ("system", agent_prompts), ("human", "{input}")
        ])
        return prompt | self.llm

    def raw_model(self):
        return self.llm


# === Unified Factory ===
class UnifiedModel:
    def __init__(self, provider: str, **kwargs):
        provider = provider.lower()
        if provider == "ollama":
            self.model = OllamaModel(**kwargs)
        elif provider == "openai":
            kwargs.setdefault("api_key", os.getenv("OPENAI_KEY"))
            if not kwargs["api_key"]:
                raise ValueError("OPENAI_API_KEY not found. Set it in .env or pass `api_key`.")
            self.model = OpenAIModel(**kwargs)
        elif provider in ["hf", "huggingface"]:
            kwargs.setdefault("hf_token", os.getenv("HF_TOKEN"))
            self.model = HFModel(**kwargs)
        elif provider == "aixplain":
            kwargs.setdefault("model_id", "640b517694bf816d35a59125")
            self.model = AiXplainModel(**kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def model_(self, agent_prompts: Optional[Text]):
        return self.model.model_(agent_prompts)

    def raw_model(self):
        return self.model.raw_model()


model_name = {
    "ollama": {"model": "llama3.2", "temperature": 0.3},
    "openai": {"model": "gpt-4o-mini", "temperature": 1.0},
    "hf": {"model": "HuggingFaceH4/zephyr-7b-beta", "temperature": 1.0},
    "aixplain": {"model": "640b517694bf816d35a59125", "temperature": 1.0},
}#.get(provider.lower())
