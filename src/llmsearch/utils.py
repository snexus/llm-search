import os
from dataclasses import dataclass
from typing import List, Optional, Union

from langchain.chains.base import Chain
# from langchain.chains.question_answering import load_qa_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.caches import BaseCache, InMemoryCache
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import PromptTemplate

# from langchain.chains import LLMChain
from langchain_core.runnables.base import RunnableSequence
from loguru import logger

from llmsearch.chroma import VectorStoreChroma
from llmsearch.config import Config, ConversrationHistorySettings, RerankerModel
from llmsearch.database.config import Base, DBSettings, get_local_session
from llmsearch.embeddings import VectorStore
from llmsearch.models.utils import get_llm
from llmsearch.ranking import BGEReranker, MarcoReranker
from llmsearch.splade import SparseEmbeddingsSplade

CHAIN_TYPE = "stuff"


@dataclass
class LLMBundle:
    """Model bundle for LLM and retriever"""

    chain: Chain
    store: VectorStore
    reranker: Optional[Union[BGEReranker, MarcoReranker]]
    sparse_search: SparseEmbeddingsSplade
    conversation_history_settings: ConversrationHistorySettings
    chunk_sizes: List[int]
    response_persist_db_settings: Optional[DBSettings] = None
    hyde_chain: Optional[RunnableSequence] = None
    hyde_enabled: bool = False
    multiquery_chain: Optional[RunnableSequence] = None
    multiquery_enabled: bool = False
    history_contextualization_chain: Optional[RunnableSequence] = None
    llm_cache: Optional[BaseCache] = None


def set_cache_folder(cache_folder_root: str):
    """Set temporary cache folder for HF models and transformers"""

    sentence_transformers_home = cache_folder_root
    transformers_cache = os.path.join(cache_folder_root, "transformers")
    hf_home = os.path.join(cache_folder_root, "hf_home")

    logger.info(
        f"Setting SENTENCE_TRANSFORMERS_HOME folder: {sentence_transformers_home}"
    )
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
    llm = get_llm(config.llm.params)  # type: ignore
    # chain = load_qa_chain(llm=llm.model, chain_type=CHAIN_TYPE, prompt=llm.prompt)
    chain = create_stuff_documents_chain(llm=llm.model, prompt=llm.prompt)

    store = VectorStoreChroma(
        persist_folder=str(config.embeddings.embeddings_path), config=config
    )
    store._load_retriever()

    reranker = None
    if config.semantic_search.reranker.enabled:
        if config.semantic_search.reranker.model == RerankerModel.BGE_RERANKER:
            reranker = BGEReranker()
        elif config.semantic_search.reranker.model == RerankerModel.MARCO_RERANKER:
            reranker = MarcoReranker()
        else:
            raise TypeError(
                "Invalid reranker type: {}", config.semantic_search.reranker.model
            )

    chunk_sizes = config.embeddings.chunk_sizes

    splade = SparseEmbeddingsSplade(config=config)
    splade.load()

    if config.persist_response_db_path is not None:
        db_settings = get_local_session(db_path=config.persist_response_db_path)
        Base.metadata.create_all(bind=db_settings.engine)
        logger.info("Initialized persistence db.")
    else:
        db_settings = None

    hyde_chain = get_hyde_chain(config, llm.model)
    multiquery_chain = get_multiquery_chain(config, llm.model)
    history_contextualization_chain = get_history_contextualize_chain(config, llm.model)

    return LLMBundle(
        chain=chain,
        reranker=reranker,
        chunk_sizes=chunk_sizes,
        sparse_search=splade,
        store=store,
        response_persist_db_settings=db_settings,
        hyde_chain=hyde_chain,
        hyde_enabled=config.semantic_search.hyde.enabled,
        multiquery_chain=multiquery_chain,
        multiquery_enabled=config.semantic_search.multiquery.enabled,
        conversation_history_settings=config.semantic_search.conversation_history_settings,
        history_contextualization_chain=history_contextualization_chain,
        llm_cache=InMemoryCache(),
    )


def get_hyde_chain(config, llm_model) -> RunnableSequence:
    logger.info("Creating HyDE chain...")
    prompt = PromptTemplate(
        template=config.semantic_search.hyde.hyde_prompt,
        input_variables=["question"],
    )
    return RunnableSequence(prompt, llm_model, StrOutputParser())
    # return RunnableSequence(
    # llm=llm_model,
    # prompt=PromptTemplate(
    # template=config.semantic_search.hyde.hyde_prompt,
    # input_variables=["question"],
    # ),
    # )


def get_multiquery_chain(config, llm_model) -> RunnableSequence:
    logger.info("Creating MultiQUery chain...")
    prompt = PromptTemplate(
        template=config.semantic_search.multiquery.multiquery_prompt,
        input_variables=["question", "n_versions"],
    )
    return RunnableSequence(prompt, llm_model, StrOutputParser())
    # return RunnableSequence(
    # llm=llm_model,
    # prompt=PromptTemplate(
    # template=config.semantic_search.multiquery.multiquery_prompt,
    # input_variables=["question", "n_versions"],
    # ),
    # )


def get_history_contextualize_chain(config: Config, llm_model) -> RunnableSequence:
    """LLM chain for history contextualization queries"""

    logger.info("Creating chat history contextualization chain...")
    prompt = PromptTemplate(
        template=config.semantic_search.conversation_history_settings.template_contextualize,
        input_variables=["chat_history", "user_question"],
    )

    return RunnableSequence(prompt, llm_model, StrOutputParser())
    # return RunnableSequence(
    # llm=llm_model,
    # prompt=PromptTemplate(
    # template=config.semantic_search.conversation_history_settings.template_contextualize,
    # input_variables=["chat_history", "user_question"],
    # ),
    # )
