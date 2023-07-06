from pydantic import BaseModel
from pathlib import Path
from typing import Optional, Union


class LlamaModelConfig(BaseModel):
    model_path: Union[Path, str]
    prompt_template: str
    model_init_params: dict = {}
    model_kwargs: dict = {}


class OpenAIModelConfig(BaseModel):
    prompt_template: str
    model_kwargs: dict = {}


class AutoGPTQModelConfig(BaseModel):
    model_folder: Union[str, Path]
    prompt_template: str
    device: str = "auto"
    tokenzier_kwargs: dict = {}
    trust_remote_code: bool = False
    use_safetensors: bool = True
    model_kwargs: dict = {}
    pipeline_kwargs: dict = {}


class HuggingFaceModelConfig(BaseModel):
    cache_folder: Optional[Union[Path, str]] = None  # will be copied from
    tokenizer_name: Optional[str] = None
    model_name: str
    prompt_template: str
    load_8bit: bool = False
    trust_remote_code: bool = False
    tokenzier_kwargs: dict = {}
    model_kwargs: dict = {}
    pipeline_kwargs: dict = {}
