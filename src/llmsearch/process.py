import statistics
import string
from typing import List, Optional

from loguru import logger

from llmsearch.config import (AppendSuffix, ObsidianAdvancedURI, ResponseModel,
                              SemanticSearchConfig, SemanticSearchOutput)
from llmsearch.utils import LLMBundle


def get_and_parse_response(llm_bundle: LLMBundle,
    query: str, 
    config: SemanticSearchConfig,
) -> ResponseModel:
    """Performs retieval augmented search (RAG).

    Args:
        llm_bundle (LLMBundle): Runtime parameters for LLM and retrievers
        config (SemanticSearchConfig): Configuration

    Returns:
        OutputModel 
    """
    
    most_relevant_docs = []
    docs = []
    
    if config.query_prefix:
        logger.info(f"Adding query prefix for retrieval: {config.query_prefix}")
        query = config.query_prefix + query
        
        
    current_reranker_score, reranker_score = -1e5, -1e5
    for retriever in llm_bundle.retrievers:
        
        # Iterate over all available chunk sizes
        for chunk_size in llm_bundle.chunk_sizes:

            # Set a filter for current chunk size or skip filter if only one chunk size is present (considerably faster)
            filter = {"chunk_size": chunk_size} if len(llm_bundle.chunk_sizes) > 1 else None
            logger.info(f"Filter: {filter}")

            res = retriever.vectorstore.similarity_search_with_relevance_scores(query, k = config.max_k, filter = filter)
            # Retrieve scores and relevant docs
            scores = [r[1] for r in res]
            relevant_docs = [r[0] for r in res]

            # Choose chunk size that is best suitable to answer the questoin
            if llm_bundle.reranker is not None:
                reranker_score, relevant_docs = llm_bundle.reranker.rerank(query, relevant_docs)
                if reranker_score > current_reranker_score:
                    docs = relevant_docs
            
            logger.info(f"Reranker median score for chunk size {chunk_size}: {reranker_score}")
            logger.info(f"Scores for chunk {chunk_size}: {scores}")
            logger.info(f"Mean score: {statistics.mean(scores)}")
    
    len_ = 0

    for doc in docs:
        doc_length = len(doc.page_content)
        if len_ + doc_length < config.max_char_size:
            most_relevant_docs.append(doc)
            len_ += doc_length
    res = llm_bundle.chain(
        {"input_documents": most_relevant_docs, "question": query},
        return_only_outputs=False,
    )

    out = ResponseModel(response=res["output_text"])
    for doc in res["input_documents"]:
        doc_name = doc.metadata["source"]
        
        for replace_setting in config.replace_output_path:
            doc_name = doc_name.replace(
                replace_setting.substring_search,
                replace_setting.substring_replace,
            )

        if config.obsidian_advanced_uri is not None:
            doc_name = process_obsidian_uri(
                doc_name, config.obsidian_advanced_uri, doc.metadata
            )

        if config.append_suffix is not None:
            doc_name = process_append_suffix(
                doc_name, config.append_suffix, doc.metadata
            )

        text = doc.page_content
        out.semantic_search.append(
            SemanticSearchOutput(
                chunk_link=doc_name, metadata=doc.metadata, chunk_text=text
            )
        )
    return out


def process_obsidian_uri(
    doc_name: str, adv_uri_config: ObsidianAdvancedURI, metadata: dict
) -> str:
    """Adds a suffix pointing to a specific heading based on the metadata supplied if doc.metadata

    Args:
        doc_name (str): Document name (partially processed, potentially)
        adv_uri_config (ObsidianAdvancedURI): contains the template to add,
                                              matches Obsidian's advanced URI plugin schem
        metadata (dict): Metadata associated with a document.

    Returns:
        str: document name with a header suffix.
    """
    print(metadata)
    append_str = adv_uri_config.append_heading_template.format(
        heading=metadata["heading"]
    )
    return doc_name + append_str


def process_append_suffix(doc_name, suffix: AppendSuffix, metadata: dict):
    fmt = PartialFormatter(missing="")
    return doc_name + fmt.format(suffix.append_template, **metadata)


class PartialFormatter(string.Formatter):
    def __init__(self, missing="~~", bad_fmt="!!"):
        self.missing, self.bad_fmt = missing, bad_fmt

    def get_field(self, field_name, args, kwargs):
        # Handle a key not found
        try:
            val = super(PartialFormatter, self).get_field(field_name, args, kwargs)
            # Python 3, 'super().get_field(field_name, args, kwargs)' works
        except (KeyError, AttributeError):
            val = None, field_name
        return val

    def format_field(self, value, spec):
        # handle an invalid format
        if value is None:
            return self.missing
        try:
            return super(PartialFormatter, self).format_field(value, spec)
        except ValueError:
            if self.bad_fmt is not None:
                return self.bad_fmt
            else:
                raise
