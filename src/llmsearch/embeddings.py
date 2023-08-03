from llmsearch.config import EmbeddingModel, EmbeddingModelType
from loguru import logger

from langchain.embeddings import HuggingFaceInstructEmbeddings, SentenceTransformerEmbeddings, HuggingFaceEmbeddings


MODELS = {
    EmbeddingModelType.instruct: HuggingFaceInstructEmbeddings,
    EmbeddingModelType.sentence_transformer: SentenceTransformerEmbeddings,
    EmbeddingModelType.huggingface: HuggingFaceEmbeddings,
}


def get_embedding_model(config: EmbeddingModel):
    """Support for additional embedding models.

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
