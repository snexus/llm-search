from pydantic import BaseModel
from pathlib import Path
from typing import Union


class LlamaModelConfig(BaseModel):
    model_path: Union[Path, str]
    prompt_template: str
    model_kwargs: dict = {}


class OpenAIModelConfig(BaseModel):
    model_name: str
    prompt_template: str
    model_kwargs: dict = {}


class AutoGPTQModelConfig(BaseModel):
    model_folder: Union[str, Path]
    prompt_template: str
    load_8bit: bool = False
    device: str = "auto"
    tokenzier_kwargs: dict = {}
    model_kwargs: dict = {}
    pipeline_kwargs: dict = {}


class HuggingFaceModelConfig(BaseModel):
    cache_folder: Union[Path, str]
    model_name: str
    prompt_template: str
    load_8bit: bool = False
    device: str = "auto"
    tokenzier_kwargs: dict = {}
    model_kwargs: dict = {}
    pipeline_kwargs: dict = {}
