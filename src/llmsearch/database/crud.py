import uuid
from sqlalchemy.orm import Session
from llmsearch.database import models
from llmsearch.config import ResponseModel, SemanticSearchOutput


def create_response(db: Session, response: ResponseModel):
    
    group_id = str(uuid.uuid4()) 
    for ss_source in response.semantic_search:
        db_source = models.Sources(
        group_id = group_id, 
        document_id = ss_source.metadata['document_id'],
        text_blob = ss_source.chunk_text, 
        text_link = ss_source.chunk_link,
        rank_score = ss_source.metadata['score']
        )
    
        db.add(db_source)
    
    db.commit()
        
    db_response_interaction = models.ResponseInteraction(
        id = str(uuid.uuid4()), 
        question_text = response.question,
        response_text = response.response,
        source_group_id = group_id
    )
    
    db.add(db_response_interaction)
    db.commit()
    db.refresh(db_response_interaction)

    
if __name__ == "__main__":
    from llmsearch.database.config import SessionLocal
    db = SessionLocal()
    
    sample_response1 = ResponseModel(
        question = "this is question 1",
        response = "this is response to question 1",
        semantic_search = [
            SemanticSearchOutput(
                chunk_link = "http://respponse1.com",
                chunk_text = "This is chunk text 1 relevant to the quesiton",
                metadata = {"score": 10.5, "document_id": "document1"}
            )       
        ]
    )
    
    
    sample_response2 = ResponseModel(
        question = "this is question 2",
        response = "this is response to question 2",
        semantic_search = [
            SemanticSearchOutput(
                chunk_link = "http://respponse2.com",
                chunk_text = "This is chunk text 2 relevant to the quesiton",
                metadata = {"score": 8.5, "document_id": "document1"}
            )       
        ]
    )
    
    create_response(db, response=sample_response1) 
    
    