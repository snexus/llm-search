import os
from dataclasses import dataclass
from typing import Any, List, Optional

from langchain.chains.base import Chain
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores.base import VectorStoreRetriever
from loguru import logger

from llmsearch.chroma import VectorStoreChroma
from llmsearch.config import Config
from llmsearch.models.utils import get_llm
from llmsearch.rerank import Reranker

CHAIN_TYPE = "stuff"


@dataclass
class LLMBundle:
    chain: Chain
    retrievers: List[VectorStoreRetriever]
    reranker: Optional[Reranker]
    chunk_sizes: List[int]


def set_cache_folder(cache_folder_root: str):
    """Set temporary cache folder for HF models and transformers"""

    sentence_transformers_home = cache_folder_root
    transformers_cache = os.path.join(cache_folder_root, "transformers")
    hf_home = os.path.join(cache_folder_root, "hf_home")

    logger.info(f"Setting SENTENCE_TRANSFORMERS_HOME folder: {sentence_transformers_home}")
    logger.info(f"Setting TRANSFORMERS_CACHE folder: {transformers_cache}")
    logger.info(f"Setting HF_HOME: {hf_home}")
    logger.info(f"Setting MODELS_CACHE_FOLDER: {cache_folder_root}")

    os.environ["SENTENCE_TRANSFORMERS_HOME"] = sentence_transformers_home
    os.environ["TRANSFORMERS_CACHE"] = transformers_cache
    os.environ["HF_HOME"] = hf_home
    os.environ["MODELS_CACHE_FOLDER"] = cache_folder_root


def get_llm_bundle(config: Config) -> LLMBundle:
    """Bundles tools needed for retrieval into auxilarry data structure

    Args:
        config (Config): instance of configuration

    Returns:
        LLMBundle:
    """

    set_cache_folder(str(config.cache_folder))
    llm = get_llm(config.llm.params)
    chain = load_qa_chain(llm=llm.model, chain_type=CHAIN_TYPE, prompt=llm.prompt)

    store = VectorStoreChroma(
        persist_folder=str(config.embeddings.embeddings_path),
        embeddings_model_config=config.embeddings.embedding_model,
    )
    embed_retriever = store.load_retriever(
        search_type=config.semantic_search.search_type, search_kwargs={"k": config.semantic_search.max_k}
    )
    reranker = Reranker() if config.semantic_search.reranker else None
    chunk_sizes = config.embeddings.chunk_sizes

    return LLMBundle(chain=chain, retrievers=[embed_retriever], reranker=reranker, chunk_sizes=chunk_sizes)
