from pathlib import Path
import shutil

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from loguru import logger

from llmsearch.nodes import get_documents_from_langchain_splitter, get_documents_from_custom_splitter
from llmsearch.parsers.markdown import markdown_splitter


class VectorStoreChroma:
    def __init__(self, persist_folder: str, hf_embed_model_name: str, chunk_size=1024, chunk_overlap=0):
        self._persist_folder = persist_folder
        self._embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
        # InstructorEmbeddingFunction(model_name="hkunlp/instructor-large")
        # HuggingFaceEmbeddings(model_name=hf_embed_model_name)
        self._splitter_conf = {"md": markdown_splitter}
        self.chunk_size = chunk_size

    def create_index_from_folder(self, folder_path: str, extension="md", clear_persist_folder=True):
        p = Path((folder_path))
        paths = list(p.glob(f"**/*.{extension}*"))

        splitter = self._splitter_conf[extension]
        # docs = get_documents_from_langchain_splitter(paths, splitter=splitter)
        docs = get_documents_from_custom_splitter(
            document_paths=paths, splitter_func=splitter, max_size=self.chunk_size
        )

        if clear_persist_folder:
            pf = Path(self._persist_folder)
            if pf.exists() and pf.is_dir():
                logger.warning(f"Deleting the content of: {pf}")
                shutil.rmtree(pf)

        # nodes = get_nodes_from_documents(document_paths=paths, chunk_parser=parser_func)

        vectordb = Chroma.from_documents(docs, self._embeddings, persist_directory=self._persist_folder)
        logger.info("Persisting the database..")
        vectordb.persist()
        vectordb = None

    def load_retriever(self, **kwargs):
        vectordb = Chroma(persist_directory=self._persist_folder, embedding_function=self._embeddings)
        retriever = vectordb.as_retriever(**kwargs)
        vectordb = None
        return retriever
