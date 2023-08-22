from abc import ABC, abstractmethod
from typing import List

from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings, SentenceTransformerEmbeddings
from loguru import logger

from llmsearch.config import Document, EmbeddingModel, EmbeddingModelType

MODELS = {
    EmbeddingModelType.instruct: HuggingFaceInstructEmbeddings,
    EmbeddingModelType.sentence_transformer: SentenceTransformerEmbeddings,
    EmbeddingModelType.huggingface: HuggingFaceEmbeddings,
}


def get_embedding_model(config: EmbeddingModel):
    """Loads an embedidng model

    Args:
        config (EmbeddingModel): Configuration for the embedding model

    Raises:
        TypeError: if model is unsupported
    """

    logger.info(f"Embedding model config: {config}")
    model_type = MODELS.get(config.type, None)

    if model_type is None:
        raise TypeError(f"Unknown model type. Got {config.type}")

    return model_type(model_name=config.model_name, **config.additional_kwargs)


class VectorStore(ABC):
    @abstractmethod
    def create_index_from_documents(self, all_docs: List[Document], clear_persist_folder: bool = True):
        pass

    @abstractmethod
    def get_documents_by_id(self, docuemt_ids: List[str]):
        pass

    @abstractmethod
    def similarity_search_with_relevance_scores(self, query: str, k: int, filter: dict):
        pass
