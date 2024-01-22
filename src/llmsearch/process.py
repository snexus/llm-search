import string
from typing import List

from loguru import logger
from llmsearch.config import (
    AppendSuffix,
    ObsidianAdvancedURI,
    ResponseModel,
    SemanticSearchOutput,
    Config,
)
from llmsearch.ranking import get_relevant_documents
from llmsearch.utils import LLMBundle
from llmsearch.database.crud import create_response


def get_and_parse_response(
    llm_bundle: LLMBundle,
    query: str,
    config: Config,
    persist_db_session=None,
    label: str = "",
) -> ResponseModel:
    """Performs retieval augmented search (RAG).

    Args:
        llm_bundle (LLMBundle): Runtime parameters for LLM and retrievers
        config (SemanticSearchConfig): Configuration

    Returns:
        OutputModel
    """

    original_query = query

    queries = []
    hyde_response = ""
    # Add HYDE queries, if required
    if llm_bundle.hyde_enabled:
        hyde_response = get_hyde_response(llm_bundle, query)
        queries += [hyde_response]

    elif llm_bundle.multiquery_enabled:
        queries += get_multiquery_response(
            llm_bundle, query, config.semantic_search.multiquery.n_versions
        )
    else:
        queries = [query]

    semantic_search_config = config.semantic_search
    most_relevant_docs, score = get_relevant_documents(
        original_query, queries, llm_bundle, semantic_search_config, label=label
    )

    res = llm_bundle.chain(
        {"input_documents": most_relevant_docs, "question": original_query},
        return_only_outputs=False,
    )

    out = ResponseModel(
        response=res["output_text"],
        question=query,
        average_score=score,
        hyde_response=hyde_response,
    )
    for doc in res["input_documents"]:
        doc_name = doc.metadata["source"]

        for replace_setting in semantic_search_config.replace_output_path:
            doc_name = doc_name.replace(
                replace_setting.substring_search,
                replace_setting.substring_replace,
            )

        if semantic_search_config.obsidian_advanced_uri is not None:
            doc_name = process_obsidian_uri(
                doc_name, semantic_search_config.obsidian_advanced_uri, doc.metadata
            )

        if semantic_search_config.append_suffix is not None:
            doc_name = process_append_suffix(
                doc_name, semantic_search_config.append_suffix, doc.metadata
            )

        text = doc.page_content
        out.semantic_search.append(
            SemanticSearchOutput(
                chunk_link=doc_name, metadata=doc.metadata, chunk_text=text
            )
        )

    if llm_bundle.response_persist_db_settings is not None:
        if persist_db_session is not None:
            session = persist_db_session
        else:
            session = llm_bundle.response_persist_db_settings.SessionLocal()
        create_response(config=config, session=session, response=out)

        # if locally created session, close it, otherwise it is assumed to be externally closed (e.g. api)
        if not persist_db_session:
            session.close()
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


def get_hyde_response(llm_bundle: LLMBundle, query: str) -> str:
    """Generates an artificial response using HyDE technique (Hypothetical Document Embeddings)
    See https://arxiv.org/pdf/2212.10496.pdf for more detail

    Args:
        llm_bundle (LLMBundle): instance of LLMBundle containing runtime parameters.
        query (str): Original query

    Returns:
        str: Artifical passage which answers the query
    """

    if llm_bundle.hyde_chain is None:
        raise TypeError("HyDE chain is not initialised. exiting.")
    res = llm_bundle.hyde_chain.run(query)
    logger.info(f"HYDE: got response: {res}")
    return res


def get_multiquery_response(
    llm_bundle: LLMBundle, query: str, n_versions: int
) -> List[str]:
    if llm_bundle.multiquery_chain is None:
        raise TypeError("MultiQuery chain is not initialised. exiting.")
    res = llm_bundle.multiquery_chain.run(question=query, n_versions=n_versions)

    logger.info(f"MultiQuery: got response: {res}")
    queries = [q.strip() for q in res.strip().split("\n") if q.strip()]
    if len(queries) != n_versions:
        raise ValueError(
            "Number of versions in multi-queries response isn't equal {}".format(
                n_versions
            )
        )
    return queries
