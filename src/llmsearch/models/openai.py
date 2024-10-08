from langchain_openai import ChatOpenAI

from llmsearch.models.abstract import AbstractLLMModel
from llmsearch.models.config import OpenAIModelConfig

# from langchain_openai import ChatOpenAI


class OpenAIModel(AbstractLLMModel):
    def __init__(self, config: OpenAIModelConfig) -> None:
        super().__init__(prompt_template=config.prompt_template)
        self.config = config

    @property
    def model(self):
        return ChatOpenAI(**self.config.model_kwargs)
