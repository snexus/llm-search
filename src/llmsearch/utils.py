import os
from dataclasses import dataclass
from typing import Any, List, Optional

from langchain.chains.base import Chain
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores.base import VectorStoreRetriever
from loguru import logger

from llmsearch.chroma import VectorStoreChroma
from llmsearch.splade import SparseEmbeddingsSplade
from llmsearch.config import Config
from llmsearch.models.utils import get_llm
from llmsearch.ranking import Reranker
from llmsearch.embeddings import VectorStore
from llmsearch.database.config import DBSettings, get_local_session, Base

CHAIN_TYPE = "stuff"


@dataclass
class LLMBundle:
    chain: Chain
    store: VectorStore
    reranker: Optional[Reranker]
    sparse_search: SparseEmbeddingsSplade
    chunk_sizes: List[int]
    response_persist_db_settings: Optional[DBSettings] = None


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
    llm = get_llm(config.llm.params) # type: ignore
    chain = load_qa_chain(llm=llm.model, chain_type=CHAIN_TYPE, prompt=llm.prompt)

    store = VectorStoreChroma(persist_folder=str(config.embeddings.embeddings_path), config=config)
    store._load_retriever()

    reranker = Reranker() if config.semantic_search.reranker else None
    chunk_sizes = config.embeddings.chunk_sizes

    splade = SparseEmbeddingsSplade(config=config)
    splade.load()

    if config.persist_response_db_path is not None:
        db_settings = get_local_session(db_path=config.persist_response_db_path)
        Base.metadata.create_all(bind = db_settings.engine)
        logger.info("Initialized persistence db.")
    else:
        db_settings = None

    return LLMBundle(
        chain=chain,
        reranker=reranker,
        chunk_sizes=chunk_sizes,
        sparse_search=splade,
        store=store,
        response_persist_db_settings=db_settings,
    )
