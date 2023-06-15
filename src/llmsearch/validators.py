from enum import Enum
from pathlib import Path
from typing import List, Union

from pydantic import BaseModel, DirectoryPath, validator
from pydantic.typing import Literal

from llmsearch.models.config import AutoGPTQModelConfig, HuggingFaceModelConfig, LlamaModelConfig, OpenAIModelConfig

models_config = {
    "llama": LlamaModelConfig,
    "openai": OpenAIModelConfig,
    "auto-gptq": AutoGPTQModelConfig,
    "huggingface": HuggingFaceModelConfig,
}


class DocumentExtension(str, Enum):
    md = "md"


class EmbeddingsConfig(BaseModel):
    doc_path: DirectoryPath
    embeddings_path: DirectoryPath
    extensions: List[DocumentExtension]


class ReplaceOutputPrefix(BaseModel):
    replace_string: str
    replace_with: str


class SemanticSearchConfig(BaseModel):
    search_type = Literal["mmr", "nearest_search"]
    replace_output_prefix: ReplaceOutputPrefix
    output_char_size: int = 2048


class LLMConfig(BaseModel):
    type: str
    params: Union[LlamaModelConfig, OpenAIModelConfig, HuggingFaceModelConfig, AutoGPTQModelConfig]

    @validator("params")
    def validate_params(cls, value, values):
        type_ = values.get("type")
        if type_ not in models_config:
            raise TypeError(f"Uknown model type {value}. Allowed types: ")

        config_type = models_config[type_]
        if not isinstance(value, config_type):
            raise TypeError(f"params should be of type: {config_type}")


class Config(BaseModel):
    cache_folder: Path
    embeddings: EmbeddingsConfig
    semantic_search: SemanticSearchConfig
    llm: LLMConfig
