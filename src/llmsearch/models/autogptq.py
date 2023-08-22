# from llmsearch.models.abstract import AbstractLLMModel
# import auto_gptq
# from llmsearch.models.config import AutoGPTQModelConfig
# import torch

# import transformers
# from langchain.llms import HuggingFacePipeline


# class AutoGPTQModel(AbstractLLMModel):
#     def __init__(self, config: AutoGPTQModelConfig) -> None:
#         super().__init__(prompt_template=config.prompt_template)
#         self.config = config

#     @property
#     def model(self):
#         if self.config.device == "auto":
#             device = (
#                 f"cuda:{torch.cuda.current_device()}"
#                 if torch.cuda.is_available()
#                 else "cpu"
#             )
#         else:
#             device = self.config.device

#         tokenizer = transformers.AutoTokenizer.from_pretrained(
#             self.config.model_folder, use_fast=True, device=device
#         )

#         model = auto_gptq.AutoGPTQForCausalLM.from_quantized(
#             self.config.model_folder,
#             use_safetensors=self.config.use_safetensors,
#             trust_remote_code=self.config.trust_remote_code,
#             device=device,
#             **self.config.model_kwargs,
#         )

#         p = transformers.pipeline(
#             "text-generation",
#             model=model,
#             tokenizer=tokenizer,
#             **self.config.pipeline_kwargs,
#         )

#         hf_pipeline = HuggingFacePipeline(pipeline=p)
#         return hf_pipeline
