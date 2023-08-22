import shutil
from pathlib import Path
from typing import List, Tuple

from langchain.vectorstores import Chroma
from loguru import logger

from llmsearch.config import Config
from llmsearch.embeddings import VectorStore, get_embedding_model
from llmsearch.parsers.splitter import Document


class VectorStoreChroma(VectorStore):
    def __init__(self, persist_folder: str, config: Config):
        self._persist_folder = persist_folder
        self._config = config
        self._embeddings = get_embedding_model(config.embeddings.embedding_model)
        self._retriever = None
        
    @property
    def retriever(self):
        if self._retriever is None:
            self._retriever = self._load_retriever()
        return self._retriever

    def create_index_from_documents(
        self,
        all_docs: List[Document],
        clear_persist_folder: bool = True,
    ):
        if clear_persist_folder:
            pf = Path(self._persist_folder)
            if pf.exists() and pf.is_dir():
                logger.warning(f"Deleting the content of: {pf}")
                shutil.rmtree(pf)

        logger.info("Generating and persisting the embeddings..")
        ids = [d.metadata["document_id"] for d in all_docs]
        vectordb = Chroma.from_documents(
            documents=all_docs, embedding=self._embeddings, ids=ids, persist_directory=self._persist_folder
        )
        vectordb.persist()

    def _load_retriever(self, **kwargs):
        vectordb = Chroma(persist_directory=self._persist_folder, embedding_function=self._embeddings)
        return vectordb.as_retriever(**kwargs)

    def get_documents_by_id(self, document_ids: List[str]) -> List[Document]:
        """Retrieves documents by ids

        Args:
            document_ids (List[str]): list of document ids

        Returns:
            List[Document]: list of documents belonging to document_ids
        """

        results = self.retriever.vectorstore.get(ids=document_ids, include=["metadatas", "documents"])
        docs = [Document(page_content=d, metadata=m) for d, m in zip(results["documents"], results["metadatas"])]
        return docs

    def similarity_search_with_relevance_scores(
        self, query: str, k: int, filter: dict
    ) -> List[Tuple[Document, float]]:

        return self.retriever.vectorstore.similarity_search_with_relevance_scores(
            query, k=self._config.semantic_search.max_k, filter=filter
        )  # type: ignore
