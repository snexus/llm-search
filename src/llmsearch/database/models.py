from datetime import datetime
import uuid

from sqlalchemy import Boolean, JSON, Column, DateTime, Float, ForeignKey, String, Text, Integer
from sqlalchemy.orm import relationship

from llmsearch.database.config import Base



def create_uuid() -> str:
    return str(uuid.uuid4())

class ResponseInteraction(Base):
    __tablename__ = "interactions"

    response_id = Column(String, primary_key=True, index=True)
    response_timestamp = Column(DateTime, default=datetime.utcnow)
    question_text = Column(Text, nullable=False)
    response_text = Column(Text, nullable=False)
    config_id = Column(String, ForeignKey("configs.config_id"))
    average_score = Column(Float, nullable=False)

    sources = relationship("InteractionSourcesBridge")
    # response_interaction = relationship("ResponseFeedback")


class InteractionSourcesBridge(Base):
    __tablename__ = "interactsourcebridge"
    
    record_id = Column(Integer, primary_key=True, autoincrement = True)
    response_id = Column(String, ForeignKey("interactions.response_id"))
    source_id = Column(String, ForeignKey("sources.source_id"))


class Sources(Base):
    __tablename__ = "sources"

    source_id = Column(String, primary_key=True, index = True)
    text_blob = Column(Text, nullable=False)
    text_link = Column(Text, nullable=False)
    rank_score = Column(Float, nullable=False)
    additional_metadata = Column(JSON)
    interactions = relationship("InteractionSourcesBridge")


class ResponseFeedback(Base):
    __tablename__ = "responses"

    feedback_id = Column(Integer, primary_key=True, autoincrement = True)
    response_id = Column(String, ForeignKey("interactions.response_id"))
    is_positive = Column(Boolean, nullable=True)
    feedback_text = Column(Text, nullable=True)


class Config(Base):
    __tablename__ = "configs"

    config_id = Column(String, primary_key=True, index = True, default = create_uuid)
    config_hash = Column(String, nullable=False)
    config_json = Column(JSON, nullable=False)
