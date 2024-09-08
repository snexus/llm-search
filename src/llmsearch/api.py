import os
from typing import Any, List

import langchain
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from sqlalchemy.orm import Session

import llmsearch.database.crud as crud
from llmsearch.config import Config, ResponseModel, get_config
from llmsearch.database.config import get_local_session
from llmsearch.process import get_and_parse_response
from llmsearch.ranking import get_relevant_documents
from llmsearch.utils import LLMBundle, get_llm_bundle

load_dotenv()
langchain.debug = True

# Load the configuration
def read_config() -> Config:
    config_file = os.environ["FASTAPI_LLM_CONFIG"]

    if not config_file:
        raise SystemError(
            "Set 'FASTAPI_LLM_CONFG' environment variable to point to a model config file."
        )

    logger.info(f"Loading configuration from {config_file}")
    config = get_config(config_file)

    return config


# Dependency: Used to get the database in our endpoints.
def get_db() -> Session:
    """Creates a database session and makes sure to close it properly."""

    if config.persist_response_db_path is None:
        raise Exception(
            "Specify database filename in `persist_response_db_path` setting in config."
        )
    db_settings = get_local_session(db_path=config.persist_response_db_path)

    db = db_settings.SessionLocal()
    try:
        yield db
    finally:
        logger.info("Closing session...")
        db.close()


def load_llm() -> LLMBundle:
    """Loads a chain to use with the api"""

    logger.info("Loading LLM...")
    bundle = get_llm_bundle(config)
    return bundle


config = read_config()
llm_bundle = load_llm()

app = FastAPI()

# Enable CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def test():
    return {"message": "Welcome to LLMSearch API"}


@app.get("/llm", response_model=ResponseModel)
async def llmsearch(
    question: str, label: str = "", db: Session = Depends(get_db)
) -> Any:  # switch to async to block execution
    if label and (label not in config.embeddings.labels):
        raise HTTPException(
            status_code=404,
            detail=f"Label '{label}' doesn't exist. Use GET /labels to get a list of labels.",
        )

    output = get_and_parse_response(
        query=question,
        llm_bundle=llm_bundle,
        config=config,
        persist_db_session=db,
        label=label,
    )
    return output.dict()


@app.get("/semantic")
async def semanticsearch(question: str):
    docs = get_relevant_documents(
        original_query=question, queries = [question], llm_bundle=llm_bundle, config=config.semantic_search, label=""
    )
    return {"sources": docs}


@app.get("/labels")
async def labels() -> List[str]:
    return config.embeddings.labels


@app.post("/feedback")
async def llmfeedback(
    response_id: str,
    is_positive: bool,
    feedback_text: str = "",
    db: Session = Depends(get_db),
):
    try:
        crud.create_feedback(
            response_id=response_id,
            session=db,
            is_positive=is_positive,
            feedback_text=feedback_text,
        )
    except crud.ResponseInteractionLookupError:
        return HTTPException(
            status_code=404,
            detail=f"Reponse interaction with id '{response_id}' doesn't exist",
        )

    return {"status": "Update complete"}


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
