from typing import Any, Dict, Generator, List, Optional

from langchain_core.callbacks import (CallbackManager,
                                      CallbackManagerForLLMRun,
                                      StreamingStdOutCallbackHandler)
from langchain_core.language_models import LLM
from llama_cpp import Llama
from loguru import logger

from llmsearch.models.abstract import AbstractLLMModel
from llmsearch.models.config import LlamaModelConfig

# class LlamaModel(AbstractLLMModel):
#     def __init__(self, config: LlamaModelConfig):
#         super().__init__(prompt_template=config.prompt_template)
#         self.config = config

#     @property
#     def model(self):

#         callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
#         llm = LlamaCpp(
#             model_path=str(self.config.model_path),
#             callback_manager=callback_manager, verbose=True, **self.config.model_kwargs
#         )
#         return llm


# values["client"] = Llama(model_path, **model_params)


class CustomLlamaLangChainModel(LLM):
    @classmethod
    def from_parameters(cls, model_path, model_init_kwargs, model_kwargs, **kwargs):
        cls.model = Llama(model_path=str(model_path), **model_init_kwargs)
        cls.model_kwargs = model_kwargs
        cls.model_path = model_path
        cls.streaming = True
        return cls(**kwargs)

    def __del__(self):
        self.model.__del__()

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        if self.streaming:
            # If streaming is enabled, we use the stream
            # method that yields as they are generated
            # and return the combined strings from the first choices's text:
            combined_text_output = ""
            for token in self.stream(prompt=prompt, stop=stop, run_manager=run_manager):
                combined_text_output += token["choices"][0]["text"]
            return combined_text_output
        else:
            result = self.model(prompt=prompt, **self.model_kwargs)
            return result["choices"][0]["text"]

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_path": self.model_path}, **self.model_kwargs}

    def stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> Generator[Dict, None, None]:
        """Yields results objects as they are generated in real time.

        BETA: this is a beta feature while we figure out the right abstraction.
        Once that happens, this interface could change.

        It also calls the callback manager's on_llm_new_token event with
        similar parameters to the OpenAI LLM class method of the same name.

        Args:
            prompt: The prompts to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            A generator representing the stream of tokens being generated.

        Yields:
            A dictionary like objects containing a string token and metadata.
            See llama-cpp-python docs and below for more.

        Example:
            .. code-block:: python

                from langchain.llms import LlamaCpp
                llm = LlamaCpp(
                    model_path="/path/to/local/model.bin",
                    temperature = 0.5
                )
                for chunk in llm.stream("Ask 'Hi, how are you?' like a pirate:'",
                        stop=["'","\n"]):
                    result = chunk["choices"][0]
                    print(result["text"], end='', flush=True)

        """
        result = self.model(prompt=prompt, stream=True, **self.model_kwargs)
        for chunk in result:
            token = chunk["choices"][0]["text"]
            log_probs = chunk["choices"][0].get("logprobs", None)
            if run_manager:
                run_manager.on_llm_new_token(
                    token=token, verbose=self.verbose, log_probs=log_probs
                )
            yield chunk


class LlamaModel(AbstractLLMModel):
    def __init__(self, config: LlamaModelConfig):
        super().__init__(prompt_template=config.prompt_template)
        self.config = config
        self._model = None

    @property
    def model(self):
        if not self._model:
            logger.info("Loading model...")
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

            logger.info("Initializing LLAmaCPP model...")
            logger.info(self.config.model_init_params)
            model_path = self.config.model_path
            model_kwargs = self.config.model_kwargs
            model_init_kwargs = self.config.model_init_params

            self._model = CustomLlamaLangChainModel.from_parameters(
                model_path=str(model_path),
                model_kwargs=model_kwargs,
                model_init_kwargs=model_init_kwargs,
                callback_manager=callback_manager,
                verbose=True,
            )
        return self._model
