from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from loguru import logger
from pydantic import BaseModel, DirectoryPath, Extra, Field, validator
from uuid import UUID, uuid4
from pydantic.typing import Literal  # type: ignore

from llmsearch.models.config import (
    HuggingFaceModelConfig,
    LlamaModelConfig,
    OpenAIModelConfig,
    AzureOpenAIModelConfig
)

models_config = {
    "llamacpp": LlamaModelConfig,
    "openai": OpenAIModelConfig,
    # "auto-gptq": AutoGPTQModelConfig,
    "huggingface": HuggingFaceModelConfig,
    "azureopenai": AzureOpenAIModelConfig 
}


def create_uuid() -> str:
    return str(uuid4())


class Document(BaseModel):
    """Interface for interacting with a document."""

    page_content: str
    metadata: dict = Field(default_factory=dict)


class DocumentExtension(str, Enum):
    md = "md"
    pdf = "pdf"
    html = "html"
    epub = "epub"
    docx = "docx"
    doc = "doc"

    
class RerankerModel(Enum):
    MARCO_RERANKER = "marco" 
    BGE_RERANKER = "bge"


class EmbeddingModelType(str, Enum):
    huggingface = "huggingface"
    instruct = "instruct"
    sentence_transformer = "sentence_transformer"


class EmbeddingModel(BaseModel):
    type: EmbeddingModelType
    model_name: str
    additional_kwargs: dict = Field(default_factory=dict)


class DocumentPathSettings(BaseModel):
    doc_path: Union[DirectoryPath, str]
    exclude_paths: List[Union[DirectoryPath, str]] = Field(default_factory=list)
    scan_extensions: List[DocumentExtension]
    additional_parser_settings: Dict[str, Any] = Field(default_factory=dict)
    passage_prefix: str = ""
    label: str = "" # Optional label, will be included in the metadata

    @validator("additional_parser_settings")
    def validate_extension(cls, value):
        for ext in value.keys():
            if ext not in DocumentExtension.__members__:
                raise TypeError(
                    f"Unknown document extension {value}. Supported: {DocumentExtension.__members__}"
                )
        return value


class EmbedddingsSpladeConfig(BaseModel):
    n_batch: int = 3


class EmbeddingsConfig(BaseModel):
    embedding_model: EmbeddingModel = EmbeddingModel(
        type=EmbeddingModelType.instruct, model_name="hkunlp/instructor-large"
    )
    embeddings_path: Union[DirectoryPath, str]
    document_settings: List[DocumentPathSettings]
    chunk_sizes: List[int] = [1024]
    splade_config: EmbedddingsSpladeConfig = EmbedddingsSpladeConfig(n_batch=5)
    
    @property
    def labels(self) -> List[str]:
        """Returns list of labels in document settings"""
        return [setting.label for setting in self.document_settings if setting.label]

    class Config:
        extra = Extra.forbid


class ObsidianAdvancedURI(BaseModel):
    append_heading_template: str


class AppendSuffix(BaseModel):
    append_template: str


class ReplaceOutputPath(BaseModel):
    substring_search: str
    substring_replace: str

class HydeSettings(BaseModel):
    enabled: bool = False
    hyde_prompt: str = "Write a short passage to answer the question: {question}"

class RerankerSettings(BaseModel):
    enabled: bool = True
    model: RerankerModel = RerankerModel.BGE_RERANKER 

class MultiQuerySettings(BaseModel):
    enabled: bool = False
    multiquery_prompt: str = """You are a helpful assistant that generates multiple questions based on the source question.
    Generate {n_versions} other questions related to: ```{question}```

    Generated questions should be separated by newlines, but shouldn't be enumerated.
    """
    # multiquery_prompt: str =  """You are an AI language model assistant. Your task is 
    # to generate {n_versions} different versions of the given user 
    # question. The questions can be generated using domain specific language to clarify the intent. Provide these alternative 
    # questions separated by newlines. Don't enumerate the alternative questions. Original question: {question}"""   
    n_versions: int = 3

class SemanticSearchConfig(BaseModel):
    search_type: Literal["mmr", "similarity"]
    replace_output_path: List[ReplaceOutputPath] = Field(default_factory=list)
    obsidian_advanced_uri: Optional[ObsidianAdvancedURI] = None
    append_suffix: Optional[AppendSuffix] = None
    reranker: RerankerSettings = RerankerSettings()
    max_k: int = 15
    max_char_size: int = 2048
    query_prefix: str = ""
    hyde: HydeSettings = HydeSettings()
    multiquery: MultiQuerySettings = MultiQuerySettings()

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


class SemanticSearchOutput(BaseModel):
    chunk_link: str
    chunk_text: str
    metadata: dict


class ResponseModel(BaseModel):
    id: UUID = Field(default_factory=create_uuid)
    question: str
    response: str
    average_score: float
    semantic_search: List[SemanticSearchOutput] = Field(default_factory=list)
    hyde_response: str = ""

class Config(BaseModel):
    cache_folder: Path
    embeddings: EmbeddingsConfig
    semantic_search: SemanticSearchConfig
    llm: LLMConfig
    persist_response_db_path: Optional[str] = None
    
        




def get_config(path: Union[str, Path]) -> Config:
    with open(path, "r") as f:
        conf_dict = yaml.safe_load(f)
    return Config(**conf_dict)
