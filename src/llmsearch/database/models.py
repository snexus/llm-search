from datetime import datetime

from sqlalchemy import Boolean, JSON, Column, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import relationship

from llmsearch.database.config import Base


class ResponseInteraction(Base):
    __tablename__ = "interactions"

    response_id = Column(String, primary_key=True, index=True)
    response_timestamp = Column(DateTime, default=datetime.utcnow)

    question_text = Column(Text, nullable=False)
    response_text = Column(Text, nullable=False)
    source_group_id = Column(String, ForeignKey("sources.group_id"))
    config_id = Column(String, ForeignKey("configs.config_id"))
    average_score = Column(Float, nullable=False)

    sources = relationship("Sources")
    # response_interaction = relationship("ResponseFeedback")


class Sources(Base):
    __tablename__ = "sources"

    source_id = Column(String, primary_key=True)
    group_id = Column(String, index=True)
    document_id = Column(String, nullable=False)
    text_blob = Column(Text, nullable=False)
    text_link = Column(Text, nullable=False)
    rank_score = Column(Float, nullable=False)
    additional_metadata = Column(JSON)


class ResponseFeedback(Base):
    __tablename__ = "responses"

    feedback_id = Column(String, primary_key=True, index=True)
    response_id = Column(String, ForeignKey("responses.response_id"))
    is_positive = Column(Boolean, nullable=True)
    feedback_text = Column(Text, nullable=True)


class Config(Base):
    __tablename__ = "configs"

    config_id = Column(String, primary_key=True)
    config_hash = Column(String, nullable=False)
    config_json = Column(JSON, nullable=False)
