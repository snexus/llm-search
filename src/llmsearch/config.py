from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import UUID, uuid4

import yaml
from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    DirectoryPath,
    Field,
    ValidationInfo,
    field_validator,
)

from llmsearch.models.config import (
    AzureOpenAIModelConfig,
    HuggingFaceModelConfig,
    LlamaModelConfig,
    OpenAIModelConfig,
)

# from pydantic.typing import Literal  # type: ignore


models_config = {
    "llamacpp": LlamaModelConfig,
    "openai": OpenAIModelConfig,
    # "auto-gptq": AutoGPTQModelConfig,
    "huggingface": HuggingFaceModelConfig,
    "azureopenai": AzureOpenAIModelConfig,
}


def create_uuid() -> str:
    return str(uuid4())


class Document(BaseModel):
    """Interface for interacting with a document."""

    page_content: str
    metadata: dict = Field(default_factory=dict)


class CustomDocumentExtension(str, Enum):
    md = "md"
    pdf = "pdf"
    docx = "docx"
    doc = "doc"


class RerankerModel(Enum):
    MARCO_RERANKER = "marco"
    BGE_RERANKER = "bge"


class PDFTableParser(str, Enum):
    GMFT = "gmft"
    AZUREDOC = "azuredoc"


class PDFImageParser(str, Enum):
    GEMINI_15_FLASH = "gemini-1.5-flash"
    GEMINI_15_PRO = "gemini-1.5-pro"


class PDFImageParseSettings(BaseModel):
    image_parser: PDFImageParser
    system_instruction: str = """You are an research assistant. You analyze the image to extract detailed information. Response must be a Markdown string in the follwing format:
- First line is a heading with image caption, starting with '# '
- Second line is empty
- From the third line on - detailed data points and related metadata, extracted from the image, in Markdown format. Don't use Markdown tables. 
"""
    user_instruction: str = (
        """From the image, extract detailed quantitative and qualitative data points."""
    )


class EmbeddingModelType(str, Enum):
    huggingface = "huggingface"
    instruct = "instruct"
    sentence_transformer = "sentence_transformer"
    openai = "openai"


class EmbeddingModel(BaseModel):
    model_config = ConfigDict()
    model_config["protected_namespaces"] = ()

    type: EmbeddingModelType
    model_name: str
    additional_kwargs: dict = Field(default_factory=dict)


class DocumentPathSettings(BaseModel):
    doc_path: Union[DirectoryPath, str]
    """Defines document folder for a given document set."""

    exclude_paths: List[Union[DirectoryPath, str]] = Field(default_factory=list)
    """List of folders to exclude from scanning."""

    scan_extensions: List[str]
    """List of extensions to scan."""

    pdf_table_parser: Optional[PDFTableParser] = None
    """If enabled, will parse tables in pdf files using a specific of a parser."""

    pdf_image_parser: Optional[PDFImageParseSettings] = None
    """If enabled, will parse images in pdf files using a specific of a parser."""

    additional_parser_settings: Dict[str, Any] = Field(default_factory=dict)
    """Optional parser settings (parser dependent)"""

    passage_prefix: str = ""

    label: str = ""  # Optional label, will be included in the metadata
    """Optional label for the document set, will be included in the metadata."""

    @field_validator("additional_parser_settings")
    def validate_extension(cls, value):
        for ext in value.keys():
            if ext not in CustomDocumentExtension.__members__:
                raise TypeError(
                    f"Custom parser settings aren't supported for document extension {value}. Supported: {CustomDocumentExtension.__members__}"
                )
        return value

    @field_validator("doc_path")
    def validate_path(cls, value):
        path = Path(value)
        if not path.exists():
            raise TypeError("Provided doc_path doesn't exist.")
        if not path.is_dir():
            raise TypeError("Provided doc_path is not a directory.")
        return value

class EmbedddingsSpladeConfig(BaseModel):
    n_batch: int = 3


class EmbeddingsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedding_model: EmbeddingModel = EmbeddingModel(
        type=EmbeddingModelType.instruct, model_name="hkunlp/instructor-large"
    )
    """Specifies embedding model to use for dense embeddings."""

    embeddings_path: Union[DirectoryPath, str]
    """Specifies output folder for embeddings."""

    document_settings: List[DocumentPathSettings]
    """Defines settings for one or more document sets."""

    chunk_sizes: List[int] = [1024]
    """List of chunk sizes for text chunking, supports multiples sizes."""

    splade_config: EmbedddingsSpladeConfig = EmbedddingsSpladeConfig(n_batch=5)
    """Specifies settings for sparse embeddings (SPLADE)."""

    @property
    def labels(self) -> List[str]:
        """Returns list of labels in document settings"""
        return [setting.label for setting in self.document_settings if setting.label]


class ObsidianAdvancedURI(BaseModel):
    append_heading_template: str


class SuffixAppend(BaseModel):
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
    Generate {n_versions} additional related questions related to: ```{question}```.
    
    Suggest only short questions without compound sentences. Suggest a variety of questions that cover different aspects of the topic.
    Make sure they are complete questions, and that they are related to the original question.

    Generated questions should be separated by newlines, but shouldn't be enumerated.
    """
    # multiquery_prompt: str =  """You are an AI language model assistant. Your task is
    # to generate {n_versions} different versions of the given user
    # question. The questions can be generated using domain specific language to clarify the intent. Provide these alternative
    # questions separated by newlines. Don't enumerate the alternative questions. Original question: {question}"""
    n_versions: int = 5


class ConversationHistoryQAPair(BaseModel):
    question: str
    answer: str


class ConversrationHistorySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False

    max_history_length: int
    """Maximum length of conversational history paris to remember (single pair = query + response)"""

    rewrite_query: bool
    """Rewrite query for better context understanding"""

    history: List[ConversationHistoryQAPair] = Field(default_factory=list)
    """Keeps history of conversation pair, up to max_history_length"""

    template_instruction: str = (
        """When answering questions, take into consideration the history of the chat converastion, which is listed below under Chat History. The chat history is in reverse chronological order, so the most recent exhange is at the top."""
    )
    template_contextualize: str = """
    Given a chat history and the latest user question \
which might reference to context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
return only reformulated question. Do NOT mention it is 'reformulated question', return only body of the question and nothing else.

    {chat_history}

    User question: {user_question}
    """

    template_header: str = "\nChat History:\n=============\n"
    template_qa_pairs: str = "User: {question}\nAssistant: {answer}\n\n"

    def add_qa_pair(self, question: str, answer: str):
        if len(self.history) == self.max_history_length:
            self.history.pop(0)

        self.history.append(ConversationHistoryQAPair(question=question, answer=answer))

    @property
    def chat_history(self) -> str:
        if not self.history:
            return ""

        out_str = self.template_header
        for qa_pair in self.history[::-1]:
            out_str += self.template_qa_pairs.format(
                question=qa_pair.question, answer=qa_pair.answer
            )

        return out_str


class SemanticSearchConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    search_type: Literal["mmr", "similarity"]
    """Configure search type, currently only similarity can be used."""

    replace_output_path: List[ReplaceOutputPath] = Field(default_factory=list)

    obsidian_advanced_uri: Optional[ObsidianAdvancedURI] = None

    append_suffix: Optional[SuffixAppend] = None
    """Allows to append suffix to document URL. Useful for deep linking to allow opening with external application, e.g. Obsidian."""

    reranker: RerankerSettings = RerankerSettings()
    """Configures re-ranker settings."""

    max_k: int = 15
    """Maximum number of documents to retrieve for dense OR sparse embedding (if using both, number of documents will be k*2)"""

    score_cutoff: Optional[float] = None
    """Documents with score less than specified will be excluded from relevant documents"""

    max_char_size: int = 16384
    """Maximum character size for query + documents to fit into context window of LLM."""

    query_prefix: str = ""
    """Prefix query with string BEFORE retrieval using embedding model."""

    hyde: HydeSettings = HydeSettings()
    """Optional configuration for HyDE."""

    multiquery: MultiQuerySettings = MultiQuerySettings()
    """Optional configuration for multi-query"""

    conversation_history_settings: ConversrationHistorySettings = (
        ConversrationHistorySettings(
            enabled=False, max_history_length=2, rewrite_query=True
        )
    )
    """Conversation history"""


class LLMConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    model_config["protected_namespaces"] = ()

    type: str
    params: dict

    @field_validator("params")
    def validate_params(cls, value, info: ValidationInfo):
        values = info.data
        type_ = values.get("type")
        if type_ not in models_config:
            raise TypeError(f"Uknown model type {value}. Allowed types: ")

        config_type = models_config[type_]  # type: ignore
        logger.info(
            f"Loading model paramaters in configuration class {config_type.__name__}"
        )
        config = config_type(
            **value
        )  # An attempt to force conversion to the required model config
        return config


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
    """Configures path to cache LLM and embedding models."""

    embeddings: EmbeddingsConfig
    """Configures document paths and embedding settings."""

    semantic_search: SemanticSearchConfig
    """Confgures semantic search settings."""

    llm: Optional[LLMConfig] = None
    "Don't use directly."

    persist_response_db_path: Optional[str] = None
    """Optional path for SQLite database for results storage."""

    def check_embeddings_exist(self) -> bool:
        """Checks if embedings exist in the specified folder"""

        p_splade = (
            Path(self.embeddings.embeddings_path) / "splade" / "splade_embeddings.npz"
        )
        p_embeddings = Path(self.embeddings.embeddings_path)
        all_parquets = list(p_embeddings.glob("*.parquet"))
        return p_splade.exists() and len(all_parquets) > 0


def get_config(path: Union[str, Path]) -> Config:
    with open(path, "r") as f:
        conf_dict = yaml.safe_load(f)
    return Config(**conf_dict)


def get_doc_with_model_config(doc_config, model_config) -> Config:
    """Loads doc and model configurations, combines, and returns an instance of Config"""

    doc_config_dict = load_yaml_file(doc_config)
    model_config_dict = load_yaml_file(model_config)

    config_dict = {**doc_config_dict, **model_config_dict}
    return Config(**config_dict)


def load_yaml_file(config) -> dict:
    """Loads YAML file or string and returns a dictionary"""
    if isinstance(config, str):
        logger.info(f"Loading doc config from a file: {config}")
        with open(config, "r", encoding="utf-8") as f:
            string_data = f.read()
    else:
        stringio = StringIO(config.getvalue().decode("utf-8"))
        string_data = stringio.read()

    config_dict = yaml.safe_load(string_data)
    return config_dict
