import shutil
from pathlib import Path
from typing import List

from langchain.vectorstores import Chroma
from loguru import logger

from llmsearch.config import EmbeddingModel
from llmsearch.embeddings import get_embedding_model, VectorStore
from llmsearch.parsers.splitter import Document


class VectorStoreChroma(VectorStore):
    def __init__(self, persist_folder: str, embeddings_model_config: EmbeddingModel):
        self._persist_folder = persist_folder
        self._embeddings = get_embedding_model(embeddings_model_config)
        self.embeddings_model_config = embeddings_model_config

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
        ids = [d.metadata['document_id'] for d in all_docs]
        vectordb = Chroma.from_documents(documents=all_docs, embedding=self._embeddings, ids = ids, persist_directory=self._persist_folder)
        vectordb.persist()

    def load_retriever(self, **kwargs):
        vectordb = Chroma(persist_directory=self._persist_folder, embedding_function=self._embeddings)
        retriever = vectordb.as_retriever(**kwargs)
        return retriever

    def get_documents_by_id(self, document_ids: List[str], retriever) -> List[Document]:
        """Retrieves documents by ids

        Args:
            document_ids (List[str]): list of document ids
            retriever (_type_): _description_

        Returns:
            List[Document]: _description_
        """
        
        results = retriever.vectorstore.get(ids = document_ids, include = ['metadatas', 'documents'])
        docs = [Document(page_content=d, metadata=m) for d,m in zip(results['documents'], results['metadatas'])]
        return docs
        
