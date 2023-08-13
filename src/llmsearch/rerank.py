from typing import List

from loguru import logger
from sentence_transformers.cross_encoder import CrossEncoder

from llmsearch.config import Document


class Reranker:
    def __init__(self, cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        logger.info("Initializing Reranker...")
        self._crdss_encoder_model_name = cross_encoder_model
        self.model = CrossEncoder(cross_encoder_model)

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        logger.info("Reranking documents ... ")
        features = [[query, doc.page_content] for doc in docs]
        scores = self.model.predict(features).tolist()
        for score, d in zip(scores, docs):
            d.metadata["score"] = score
        print(scores)
        return [doc for doc in sorted(docs, key=lambda d: d.metadata["score"], reverse=True)]
