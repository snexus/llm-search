from enum import Enum
from pathlib import Path
from typing import List, Union

import yaml
from pydantic import BaseModel, DirectoryPath, validator, Extra
from pydantic.typing import Literal  # type: ignore
from loguru import logger

from llmsearch.models.config import AutoGPTQModelConfig, HuggingFaceModelConfig, LlamaModelConfig, OpenAIModelConfig

models_config = {
    "llamacpp": LlamaModelConfig,
    "openai": OpenAIModelConfig,
    "auto-gptq": AutoGPTQModelConfig,
    "huggingface": HuggingFaceModelConfig,
}


class DocumentExtension(str, Enum):
    md = "md"


class EmbeddingsConfig(BaseModel):
    doc_path: DirectoryPath
    embeddings_path: DirectoryPath
    scan_extension: DocumentExtension

    class Config:
        extra = Extra.forbid


class ReplaceOutputPath(BaseModel):
    substring_search: str
    substring_replace: str


class SemanticSearchConfig(BaseModel):
    search_type: Literal["mmr", "nearest_search"]
    replace_output_path: ReplaceOutputPath
    max_char_size: int = 2048

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.forbid


class LLMConfig(BaseModel):
    type: str
    params: dict  # Union[LlamaModelConfig, OpenAIModelConfig, HuggingFaceModelConfig, AutoGPTQModelConfig]

    @validator("params")
    def validate_params(cls, value, values):
        type_ = values.get("type")
        if type_ not in models_config:
            raise TypeError(f"Uknown model type {value}. Allowed types: ")

        config_type = models_config[type_]
        logger.info(f"Loading model paramaters in configuration class {config_type.__name__}")
        config = config_type(**value)  # An attempt to force conversion to the required model config
        return config

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.forbid


class Config(BaseModel):
    cache_folder: Path
    embeddings: EmbeddingsConfig
    semantic_search: SemanticSearchConfig
    llm: LLMConfig


def get_config(path: Union[str, Path]) -> Config:
    with open(path, "r") as f:
        conf_dict = yaml.safe_load(f)
    return Config(**conf_dict)
