from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict


class LlamaModelConfig(BaseModel):
    model_config = ConfigDict()
    model_config["protected_namespaces"] = ()

    model_path: Union[Path, str]
    prompt_template: str
    model_init_params: dict = {}
    model_kwargs: dict = {}


class OpenAIModelConfig(BaseModel):
    model_config = ConfigDict()
    model_config["protected_namespaces"] = ()

    prompt_template: str
    model_kwargs: dict = {}


class AzureOpenAIModelConfig(BaseModel):
    model_config = ConfigDict()
    model_config["protected_namespaces"] = ()

    prompt_template: str
    deployment_name: str
    model_name: str
    model_kwargs: dict = {}
    openai_api_type: str = "azure"
    openai_api_version: str = "2023-05-15"
    openai_api_base: str


# class AutoGPTQModelConfig(BaseModel):
#     model_folder: Union[str, Path]
#     prompt_template: str
#     device: str = "auto"
#     tokenzier_kwargs: dict = {}
#     trust_remote_code: bool = False
#     use_safetensors: bool = True
#     model_kwargs: dict = {}
#     pipeline_kwargs: dict = {}


class HuggingFaceModelConfig(BaseModel):
    model_config = ConfigDict()
    model_config["protected_namespaces"] = ()

    cache_folder: Optional[Union[Path, str]] = None  # will be copied from
    tokenizer_name: Optional[str] = None
    model_name: str
    prompt_template: str
    load_8bit: bool = False
    trust_remote_code: bool = False
    tokenzier_kwargs: dict = {}
    model_kwargs: dict = {}
    pipeline_kwargs: dict = {}
