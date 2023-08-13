import os
import uvicorn
from collections import namedtuple

from fastapi import FastAPI
from langchain.chains.question_answering import load_qa_chain
from loguru import logger

from llmsearch.chroma import VectorStoreChroma
from llmsearch.cli import set_cache_folder
from llmsearch.config import get_config
from llmsearch.models.utils import get_llm
from llmsearch.process import get_and_parse_response

LLMParams = namedtuple("LLMParams", "chain embed_retriever config")


def load_llm(k : int = 7) -> LLMParams:
    """Loads a chain to use with the api

    Args:
        k (int, optional): Number of documents to retrieve from Vector store. Defaults to 7.

    Raises:
        SystemError: If `FASTAPI_LLM_CONFIG` is not set

    Returns:
        LLMParams: An instance of LLMParams Langchain's chain, embedding retriever and an instance of configuration
    """

    config_file = os.environ["FASTAPI_LLM_CONFIG"]

    if not config_file:
        raise SystemError(
            "Set 'FASTAPI_LLM_CONFG' environment variable to point to a model config file."
        )
    logger.info("Loading LLM...")
    config = get_config(config_file)
    set_cache_folder(str(config.cache_folder))

    llm = get_llm(config.llm.params)  # type: ignore
    print(llm)
    store = VectorStoreChroma(persist_folder=str(config.embeddings.embeddings_path), embeddings_model_config=config.embeddings.embedding_model)
    embed_retriever = store.load_retriever(
        search_type=config.semantic_search.search_type, search_kwargs={"k": k}
    )

    chain = load_qa_chain(llm=llm.model, chain_type="stuff", prompt=llm.prompt)
    llm_params = LLMParams(chain=chain, embed_retriever=embed_retriever, config=config)

    return llm_params


llm_params = load_llm()

app = FastAPI()


@app.get("/test")
def test():
    return {"Hello": "world"}


@app.get("/llm")
def llmsearch(question: str):
    output = get_and_parse_response(
        query=question,
        chain=llm_params.chain,
        retrievers=llm_params.embed_retriever,
        config=llm_params.config.semantic_search,
    )
    
    return output.json(indent=2)

@app.get("/semantic")
def semanticsearch(question: str):
    docs = llm_params.embed_retriever.get_relevant_documents(query=question)
    return {"sources": docs}

    
def main():
    uvicorn.run(app)