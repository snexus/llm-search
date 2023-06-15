from llmsearch.models.abstract import AbstractLLMModel
from llmsearch.models.config import HuggingFaceModelConfig

import torch
import transformers
from langchain.llms import HuggingFacePipeline

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextGenerationPipeline,
    pipeline,
)


class HuggingFaceModel(AbstractLLMModel):
    def __init__(self, config: HuggingFaceModelConfig) -> None:
        super().__init__(prompt_template=config.prompt_template)
        self.config = config

    @property
    def model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, **self.config.tokenzier_kwargs)
        model_kwargs = self.confg.model_kwargs

        model_ = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            cache_dir=self.cache_folder,
            device_map="auto",
            load_in_8bit=self.load_8bit,
        )
        
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_,
            tokenizer=tokenizer,
            eos_token_id=tokenizer.eos_token_id,
            torch_dtype=torch.bfloat16,
            model_kwargs=model_kwargs,
            **self.config.pipeline_kwargs
        )

        llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs=model_kwargs)
        return llm
