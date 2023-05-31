from typing import Any, List, Mapping, Optional
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms.base import LLM
from llama_index import LangchainEmbedding, LLMPredictor, PromptHelper, ServiceContext
from transformers import pipeline
import torch

load_dotenv()


class AbstractModel:
    def __init__(self, embedding_model_name: str, 
                 llm_model_name: str, 
                 max_input_size: int, 
                 num_output: int,
                 max_chunk_overlap: int = 20,
                 cache_folder=None):
        self._llm_model_name = llm_model_name
        self._embedding_model_name = embedding_model_name
        self._service_context = None
        self._cache_folder = cache_folder
        
        self._max_input_size = max_input_size
        self._num_output = num_output
        self._max_chunk_overlap = max_chunk_overlap
        
    @property
    def prompt_helper(self) -> PromptHelper:
        return PromptHelper(max_input_size = self._max_input_size, num_output = self._num_output, max_chunk_overlap = self._max_chunk_overlap)


class LLMOpenAIWrapper(AbstractModel):
    def __init__(self, temperature: int = 0, **kwargs):
        super().__init__(**kwargs)
        self._temperature = temperature
        
    @property
    def service_context(self):
        if self._service_context is None:
            embed_model = LangchainEmbedding(
                HuggingFaceEmbeddings(model_name=self._embedding_model_name) 
            )
            llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=self._temperature, model_name=self._llm_model_name))

            self._service_context = ServiceContext.from_defaults(
                llm_predictor=llm_predictor, embed_model=embed_model, chunk_size_limit=512, prompt_helper=self.prompt_helper
            )
        return self._service_context


class CustomLLM(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.pipeline(prompt)
        return response[0]["generated_text"] # Hardcoded for the time being

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "custom"

    @classmethod
    def from_custom_model(cls, model_name: str, model_kwargs):
        cls.model_name = model_name
        cls.pipeline = pipeline(
            model=model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            model_kwargs=model_kwargs,
        )
        return cls()


class HuggingFaceWrapper(AbstractModel):
    @property
    def service_context(self):
        if self._service_context is None:
            embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name=self._embedding_model_name))

            llm_predictor = LLMPredictor(
                llm=CustomLLM.from_custom_model(
                    model_name=self._llm_model_name, model_kwargs={"cache_dir": self._cache_folder}
                )
            )

            self._service_context = ServiceContext.from_defaults(
                llm_predictor=llm_predictor, embed_model=embed_model, chunk_size_limit=512, prompt_helper=self.prompt_helper
            )
        return self._service_context
