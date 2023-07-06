import torch
import transformers
from langchain.llms import HuggingFacePipeline

from llmsearch.models.abstract import AbstractLLMModel
from llmsearch.models.config import HuggingFaceModelConfig


class HuggingFaceModel(AbstractLLMModel):
    def __init__(self, config: HuggingFaceModelConfig) -> None:
        super().__init__(prompt_template=config.prompt_template)
        self.config = config

    @property
    def model(self):
        tokenizer_name = (
            self.config.tokenizer_name
            if self.config.tokenizer_name is not None
            else self.config.model_name
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_name, **self.config.tokenzier_kwargs
        )
        # model_kwargs = self.config.model_kwargs.update({"cache_dir": self.config.cache_folder})

        model_ = transformers.AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            cache_dir=self.config.cache_folder,
            device_map="auto",
            trust_remote_code=self.config.trust_remote_code,
            load_in_8bit=self.config.load_8bit,
            **self.config.model_kwargs
        )

        pipeline = transformers.pipeline(
            "text-generation",
            model=model_,
            tokenizer=tokenizer,
            eos_token_id=tokenizer.eos_token_id,
            torch_dtype=torch.bfloat16,
            model_kwargs=self.config.model_kwargs,
            **self.config.pipeline_kwargs
        )

        llm = HuggingFacePipeline(pipeline=pipeline)
        return llm
