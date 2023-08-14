import os
import asyncio
from typing import Tuple
from dotenv import load_dotenv

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import langchain

from llmsearch.config import Config, get_config
from llmsearch.process import get_and_parse_response
from llmsearch.utils import LLMBundle, get_llm_bundle

load_dotenv()
langchain.debug = True

def load_llm() -> Tuple[LLMBundle, Config]:
    """Loads a chain to use with the api

    Args:
        k (int, optional): Number of documents to retrieve from Vector store. Defaults to 7.

    Raises:
        SystemError: If `FASTAPI_LLM_CONFIG` is not set
    """

    config_file = os.environ["FASTAPI_LLM_CONFIG"]

    if not config_file:
        raise SystemError("Set 'FASTAPI_LLM_CONFG' environment variable to point to a model config file.")
    logger.info("Loading LLM...")

    config = get_config(config_file)
    bundle = get_llm_bundle(config)

    return bundle, config


llm_bundle, config = load_llm()

app = FastAPI()

# Enable CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/test")
def test():
    return {"Hello": "world"}


@app.get("/llm")
async def llmsearch(question: str): # switch to async to block execution
    output = get_and_parse_response(
        query=question,
        chain=llm_bundle.chain,
        retrievers=llm_bundle.retrievers,
        config=config.semantic_search,
        reranker=llm_bundle.reranker,
    )
    return output.json(indent=2)


@app.get("/semantic")
async def semanticsearch(question: str):
    docs = llm_bundle.retrievers[0].get_relevant_documents(query=question)
    return {"sources": docs}


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
