import hashlib
import uuid

from loguru import logger
from sqlalchemy.orm import Session

from llmsearch.config import Config, ResponseModel
from llmsearch.database import models


class ResponseInteractionLookupError(Exception):
    """Raised in case response with a given response id doesn't exist"""


def get_or_store_config(config: Config, session: Session) -> str:
    config_json = config.json()
    config_hash = hashlib.md5(config_json.encode()).hexdigest()
    print(config_hash)

    config_id = (
        session.query(models.Config.config_id)
        .filter(models.Config.config_hash == config_hash)
        .scalar()
    )
    if config_id is None:
        logger.warning("Config doesn't exist in the database, creating new entry...")

        config_id = str(uuid.uuid4())

        db_config = models.Config(
            config_id=config_id, config_hash=config_hash, config_json=config_json
        )
        session.add(db_config)
        session.commit()
    logger.info(f"Retrieved config id: {config_id}")
    return config_id


def create_response(
    config: Config, session: Session, response: ResponseModel
) -> models.ResponseInteraction:
    config_id = get_or_store_config(config, session)

    sources = []
    for ss_source in response.semantic_search:
        sources.append(
            models.Sources(
                source_id=models.create_uuid(),
                text_blob=ss_source.chunk_text,
                text_link=ss_source.chunk_link,
                rank_score=ss_source.metadata["score"],
                additional_metadata=ss_source.metadata,
            )
        )

    db_response_interaction = models.ResponseInteraction(
        response_id=response.id,
        question_text=response.question,
        response_text=response.response,
        average_score=response.average_score,
        config_id=config_id,
    )

    # Add bridge records
    bridges = []
    for s in sources:
        session.add(s)
        session.add(
            models.InteractionSourcesBridge(
                response_id=db_response_interaction.response_id, source_id=s.source_id
            )
        )

    session.add(db_response_interaction)
    session.add_all(bridges)
    session.add_all(sources)
    session.commit()
    logger.info(
        f"Saved response to the database. Response id: {db_response_interaction.response_id}"
    )
    return db_response_interaction


def create_feedback(
    session: Session, response_id: str, is_positive: bool, feedback_text: str = ""
):
    logger.info(
        f"Updating response {response_id} with values: is_positive = {is_positive}, feedback_text = '{feedback_text}'"
    )
    resp_int = session.query(models.ResponseInteraction).filter(
        models.ResponseInteraction.response_id == response_id
    )
    if resp_int.scalar() is None:
        raise ResponseInteractionLookupError(
            f"Response id {response_id} doesn't exist."
        )

    resp_int.update({"is_positive": is_positive, "feedback_text": feedback_text})
    session.commit()
