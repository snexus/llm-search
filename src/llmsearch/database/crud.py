import uuid
from sqlalchemy.orm import Session
from llmsearch.database.config import Base, DBSettings
from loguru import logger
from llmsearch.database import models
from llmsearch.config import ResponseModel, SemanticSearchOutput, Config
import hashlib


def get_or_store_config(config: Config, session: Session) -> str:
    config_json = config.json()
    config_hash = hashlib.md5(config_json.encode()).hexdigest()

    config_id = (
        session.query(models.Config.config_id).filter(models.Config.config_hash == config_hash).scalar()
    )
    if config_id is None:
        logger.warning("Config doesn't exist in the database, creating new entry...")

        config_id = str(uuid.uuid4())

        db_config = models.Config(config_id=config_id, config_hash=config_hash, config_json=config_json)
        session.add(db_config)
        session.commit()
    logger.info(f"Retrieved config id: {config_id}")
    return config_id


def create_response(config: Config, session: Session, response: ResponseModel) -> models.ResponseInteraction:
    config_id = get_or_store_config(config, session)

    group_id = str(uuid.uuid4())
    for ss_source in response.semantic_search:
        db_source = models.Sources(
            source_id=str(uuid.uuid4()),
            group_id=group_id,
            document_id=ss_source.metadata["document_id"],
            text_blob=ss_source.chunk_text,
            text_link=ss_source.chunk_link,
            rank_score=ss_source.metadata["score"],
            additional_metadata=ss_source.metadata,
        )

        session.add(db_source)

    session.commit()

    db_response_interaction = models.ResponseInteraction(
        response_id=str(uuid.uuid4()),
        question_text=response.question,
        response_text=response.response,
        average_score=response.average_score,
        source_group_id=group_id,
        config_id=config_id,
    )

    session.add(db_response_interaction)
    session.commit()
    logger.info(f"Saved response to the database. Response id: {db_response_interaction.response_id}")
    return db_response_interaction

def create_feedback(session: Session, response_id: str, is_positive: bool, feedback_text: str = ""):
    db_feedback = models.ResponseFeedback(
        feedback_id=str(uuid.uuid4()),
        response_id=response_id,
        is_positive=is_positive,
        feedback_text=feedback_text,
    )
    session.add(db_feedback)
    session.commit()



if __name__ == "__main__":
    from llmsearch.database.config import get_local_session, Base
    from llmsearch.config import get_config

    db_settings = get_local_session("responses2.db")
    session = db_settings.SessionLocal()
    # db = SessionLocal()

    #     Base.metadata.create_all(bind = engine)

    sample_response1 = ResponseModel(
        question="this is question 1",
        response="this is response to question 1",
        average_score=5.25,
        semantic_search=[
            SemanticSearchOutput(
                chunk_link="http://respponse1.com",
                chunk_text="This is chunk text 1 relevant to the quesiton",
                metadata={"score": 10.5, "document_id": "document1"},
            ),
            SemanticSearchOutput(
                chunk_link="http://respponse2.com",
                chunk_text="This is chunk text 2 relevant to the quesiton",
                metadata={"score": 8.5, "document_id": "document1"},
            ),
        ],
    )

    sample_response2 = ResponseModel(
        question="this is question 2",
        response="this is response to question 2",
        average_score=5.25,
        semantic_search=[
            SemanticSearchOutput(
                chunk_link="http://respponse2.com",
                chunk_text="This is chunk text 2 relevant to the quesiton",
                metadata={"score": 8.5, "document_id": "document1"},
            )
        ],
    )

    config = get_config(path="sample_templates/obsidian_conf.yaml")
    create_response(session, config, response=sample_response1)
    resp = create_response(session, config, response=sample_response2)
    create_feedback(session, resp.response_id, is_positive=False)

    # config_id = get_or_store_config(db, config)
    # response_interaction  = create_response(db, config_id = config_id, response=sample_response1)
    # create_feedback(db, response_id=response_interaction.response_id, is_positive=True)
