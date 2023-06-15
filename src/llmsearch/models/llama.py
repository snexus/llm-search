from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp

from llmsearch.models.abstract import AbstractLLMModel
from llmsearch.models.config import LlamaModelConfig


class LlamaModel(AbstractLLMModel):
    def __inot__(self, config: LlamaModelConfig):
        super().__init__(prompt_template=config.prompt_template)
        self.config = config

    @property
    def model(self):
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        llm = LlamaCpp(
            model_path=self.config.model_path, 
            callback_manager=callback_manager, verbose=True, **self.confg.model_kwargs
        )
        return llm
