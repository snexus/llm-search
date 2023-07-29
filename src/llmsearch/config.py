from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from loguru import logger
from pydantic import BaseModel, DirectoryPath, Extra, Field, validator
from pydantic.typing import Literal  # type: ignore

from llmsearch.models.config import (AutoGPTQModelConfig,
                                     HuggingFaceModelConfig, LlamaModelConfig,
                                     OpenAIModelConfig)

models_config = {
    "llamacpp": LlamaModelConfig,
    "openai": OpenAIModelConfig,
    "auto-gptq": AutoGPTQModelConfig,
    "huggingface": HuggingFaceModelConfig,
}


class Document(BaseModel):
    """Interface for interacting with a document."""

    page_content: str
    metadata: dict = Field(default_factory=dict)
    
    
class DocumentExtension(str, Enum):
    md = "md"
    pdf = "pdf"
    html = "html"
    epub = "epub"
    
class DocumentPathSettings(BaseModel):
    doc_path: DirectoryPath
    exclude_paths: List[DirectoryPath] = Field(default_factory=list)
    scan_extensions: List[DocumentExtension]
    additional_parser_settings: Dict[str, Any] = Field(default_factory=dict)
    chunk_size: int = 1024

    @validator("additional_parser_settings")
    def validate_extension(cls, value):
        for ext in value.keys(): 
            if ext not in  DocumentExtension.__members__:
                raise TypeError(f"Unknown document extension {value}. Supported: {DocumentExtension.__members__}")
        return value



class EmbeddingsConfig(BaseModel):
    embeddings_path: DirectoryPath
    document_settings: List[DocumentPathSettings]

    class Config:
        extra = Extra.forbid


class ObsidianAdvancedURI(BaseModel):
    append_heading_template: str


class AppendSuffix(BaseModel):
    append_template: str


class ReplaceOutputPath(BaseModel):
    substring_search: str
    substring_replace: str


class SemanticSearchConfig(BaseModel):
    search_type: Literal["mmr", "similarity"]
    replace_output_path: ReplaceOutputPath
    obsidian_advanced_uri: Optional[ObsidianAdvancedURI] = None
    append_suffix: Optional[AppendSuffix] = None
    max_char_size: int = 2048

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.forbid


class LLMConfig(BaseModel):
    type: str
    params: dict

    @validator("params")
    def validate_params(cls, value, values):
        type_ = values.get("type")
        if type_ not in models_config:
            raise TypeError(f"Uknown model type {value}. Allowed types: ")

        config_type = models_config[type_]
        logger.info(
            f"Loading model paramaters in configuration class {config_type.__name__}"
        )
        config = config_type(
            **value
        )  # An attempt to force conversion to the required model config
        return config

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.forbid


class Config(BaseModel):
    cache_folder: Path
    embeddings: EmbeddingsConfig
    semantic_search: SemanticSearchConfig
    llm: LLMConfig


class SemanticSearchOutput(BaseModel):
    chunk_link: str
    chunk_text: str
    metadata: dict

class ResponseModel(BaseModel):
    response: str
    semantic_search: List[SemanticSearchOutput] = Field(default_factory=list)


def get_config(path: Union[str, Path]) -> Config:
    with open(path, "r") as f:
        conf_dict = yaml.safe_load(f)
    return Config(**conf_dict)
