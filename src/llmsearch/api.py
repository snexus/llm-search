"""FastAPI server for LLMSearch."""

import os
import gc

import torch


# This is a temporary solution due to incompatimbility of ChromaDB with latest version of Protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from llmsearch.chroma import VectorStoreChroma
from functools import lru_cache
from typing import Any, List

import langchain
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi_mcp import FastApiMCP
from loguru import logger

# import llmsearch.database.crud as crud
from llmsearch.config import Config, ResponseModel, get_doc_with_model_config

# from llmsearch.database.config import get_local_session
from llmsearch.process import get_and_parse_response
from llmsearch.ranking import get_relevant_documents
from llmsearch.utils import LLMBundle, get_llm_bundle

from llmsearch.embeddings import (
    EmbeddingsHashNotExistError,
    update_embeddings,
)

# from sqlalchemy.orm import Session


load_dotenv()
langchain.debug = False


# Load the configuration
def read_config() -> Config:
    """Reads the configuration from environment variables and config files."""
    rag_config_file = os.environ["FASTAPI_RAG_CONFIG"]
    llm_config_file = os.environ["FASTAPI_LLM_CONFIG"]

    if not rag_config_file or not llm_config_file:
        raise SystemError(
            "Set 'FASTAPI_RAG_CONFIG' and 'FASTAPI_LLM_CONFIG' environment variable to point to a model config file."
        )

    logger.info(f"Loading configuration from {rag_config_file}")
    conf = get_doc_with_model_config(rag_config_file, llm_config_file)
    return conf


# Cache the configuration
@lru_cache()
def get_config() -> Config:
    """Loads and caches the configuration."""
    return read_config()


# Cache the LLM bundle
@lru_cache()
def get_cached_llm_bundle() -> LLMBundle:
    """Loads and caches the LLM bundle."""
    config = get_config()
    logger.info("Loading LLM...")
    bundle = get_llm_bundle(config)
    return bundle
    # return load_llm()


# Dependency for injecting the LLM bundle
def get_llm_bundle_cached() -> LLMBundle:
    """Provides the cached LLM bundle."""
    return get_cached_llm_bundle()


# Dependency: Used to get the database in our endpoints.
# def get_db() -> Session:
#     """Creates a database session and makes sure to close it properly."""

#     if semantic_search_conf.persist_response_db_path is None:
#         raise Exception("Specify database filename in `persist_response_db_path` setting in config.")
#     db_settings = get_local_session(db_path=get_config().persist_response_db_path)

#     db = db_settings.SessionLocal()
#     try:
#         yield db
#     finally:
#         logger.info("Closing session...")
#         db.close()


# def load_llm() -> LLMBundle:
#     """Loads a chain to use with the api"""

#     logger.info("Loading LLM...")
#     bundle = get_llm_bundle(config)
#     return bundle


# config = read_config()
# llm_bundle = load_llm()

api_app = FastAPI()
mcp = FastApiMCP(
    api_app,
    name="pyLLMSearch MCP Server",
    description="pyLLMSearch MCP Server",
    describe_all_responses=True,  # Include all possible response schemas
    describe_full_response_schema=True,  # Include full JSON schema in descriptions
    include_operations=["rag_retrieve_chunks", "rag_generate_answer", "rag_generate_answer_simple", "rag_update_index"],
)


mcp.mount()


# Enable CORS
origins = ["*"]
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@api_app.get("/")
def test():
    """Test endpoint to check if the API is running."""
    return {"message": "Welcome to LLMSearch API"}


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


def unload_model(llm_bundle: LLMBundle):
    """Unloads llm_bundle from the state to free up the GPU memory"""

    llm_bundle.store = None  # type: ignore
    llm_bundle.chain = None  # type: ignore
    llm_bundle.reranker = None
    llm_bundle.hyde_chain = None
    llm_bundle.multiquery_chain = None

    get_cached_llm_bundle.cache_clear()
    gc.collect()

    llm_bundle = None  # type: ignore
    gc.collect()

    with torch.no_grad():
        torch.cuda.empty_cache()


@api_app.put("/update", operation_id="rag_update_index")
def update_index(llm_bundle: LLMBundle = Depends(get_llm_bundle_cached), 
                 config: Config = Depends(get_config)) -> Any:
    """Updates the index with the latest documents."""

    set_cache_folder(str(config.cache_folder))

    vs = VectorStoreChroma(persist_folder=str(config.embeddings.embeddings_path), config=config)
    try:
        logger.debug("Updating embeddings")
        stats = update_embeddings(config, vs)
    except EmbeddingsHashNotExistError as exc:
        raise HTTPException(
            status_code=500,
            detail="Couldn't find hash files. Please re-create the index using current version of the app.",
        ) from exc
    else:
        return stats
    finally:
        logger.debug("Cleaning memory and re-Loading model...")

        vs.unload()

        vs = None  # type: ignore

        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()

        unload_model(llm_bundle)


@api_app.get("/llm", response_model=ResponseModel, operation_id="rag_generate_answer")
async def llmsearch(
    question: str,
    label: str = "",
    llm_bundle: LLMBundle = Depends(get_llm_bundle_cached),
) -> Any:  # switch to async to block execution
    """Retrieves answer to the question from the embedded documents, using semantic search."""
    if label and (label not in get_config().embeddings.labels):
        raise HTTPException(
            status_code=404,
            detail=f"Label '{label}' doesn't exist. Use GET /labels to get a list of labels.",
        )

    output = get_and_parse_response(
        query=question,
        llm_bundle=llm_bundle,
        config=get_config(),
        label=label,
    )
    return output.model_dump()


@api_app.get("/rag_text_response", operation_id="rag_generate_answer_simple")
async def llmsearch_simple(
    question: str,
    label: str = "",
    llm_bundle: LLMBundle = Depends(get_llm_bundle_cached),
) -> str:
    """Retrieves answer to the question from the embedded documents, using semantic search."""
    if label and (label not in get_config().embeddings.labels):
        raise HTTPException(
            status_code=404,
            detail=f"Label '{label}' doesn't exist. Use GET /labels to get a list of labels.",
        )

    output = get_and_parse_response(
        query=question,
        llm_bundle=llm_bundle,
        config=get_config(),
        label=label,
    )
    return output.response


@api_app.get("/rag_chunks", operation_id="rag_retrieve_chunks")
async def semanticsearch(question: str):
    """Retrieves chunks of information relevant to the question from the embedded documents, using semantic search."""
    docs = get_relevant_documents(
        original_query=question,
        queries=[question],
        llm_bundle=get_llm_bundle_cached(),
        config=get_config().semantic_search,
        label="",
    )
    return {"sources": docs}


@api_app.get("/labels")
async def labels() -> List[str]:
    """Returns a list of labels for the embeddings."""
    return get_config().embeddings.labels


# @api_app.post("/feedback")
# async def llmfeedback(
#     response_id: str,
#     is_positive: bool,
#     feedback_text: str = "",
#     db: Session = Depends(get_db),
# ):
#     """Saves feedback for a given response id."""
#     try:
#         crud.create_feedback(
#             response_id=response_id,
#             session=db,
#             is_positive=is_positive,
#             feedback_text=feedback_text,
#         )
#     except crud.ResponseInteractionLookupError:
#         return HTTPException(
#             status_code=404,
#             detail=f"Reponse interaction with id '{response_id}' doesn't exist",
#         )

#     return {"status": "Update complete"}


def main():
    """Main function to run the FastAPI app."""
    uvicorn.run(api_app, host="0.0.0.0", port=8000)


# Refresh mcp server
mcp.setup_server()

if __name__ == "__main__":
    main()
