import os

from langchain_openai import AzureChatOpenAI

from llmsearch.models.abstract import AbstractLLMModel
from llmsearch.models.config import AzureOpenAIModelConfig


class AzureOpenAIModel(AbstractLLMModel):
    def __init__(self, config: AzureOpenAIModelConfig) -> None:
        super().__init__(prompt_template=config.prompt_template)
        self.config = config

    @property
    def model(self):
        os.environ["OPENAI_API_TYPE"] = self.config.openai_api_type
        os.environ["OPENAI_API_BASE"] = self.config.openai_api_base
        os.environ["OPENAI_API_VERSION"] = self.config.openai_api_version

        return AzureChatOpenAI(
            deployment_name=self.config.deployment_name,
            model=self.config.model_name,
            **self.config.model_kwargs
        )
