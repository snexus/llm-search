import os
from pathlib import Path
import shutil


from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from loguru import logger

from llmsearch.nodes import get_documents_from_langchain_splitter, get_documents_from_custom_splitter
from llmsearch.parsers.markdown import markdown_splitter


class VectorStoreChroma:
    def __init__(
        self, persist_folder: str, hf_embed_model_name: str, chunk_size=1024, chunk_overlap=0
    ):
        self._persist_folder = persist_folder
        self._embeddings = HuggingFaceEmbeddings(model_name=hf_embed_model_name)
        # self._splitter_conf = {
        #     "md": RecursiveCharacterTextSplitter.from_language(
        #         language=Language.MARKDOWN, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        #     )
        # }
        
        self._splitter_conf = {
            "md": markdown_splitter
        }
        self.chunk_size = chunk_size
        

    def create_index_from_folder(self, folder_path: str, extension="md", clear_persist_folder = True):
        p = Path((folder_path))
        paths = list(p.glob(f"**/*.{extension}*"))

        splitter = self._splitter_conf[extension]
        #docs = get_documents_from_langchain_splitter(paths, splitter=splitter)
        docs = get_documents_from_custom_splitter(document_paths=paths, splitter_func=splitter, max_size=self.chunk_size)
        
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
        retriever =  vectordb.as_retriever(**kwargs)
        vectordb = None
        return retriever


if __name__ == "__main__":
    STORAGE_FOLDER_ROOT = "/storage/llm/"
    CACHE_FOLDER_ROOT = os.path.join(STORAGE_FOLDER_ROOT, "cache")
    EMBEDDINGS_PERSIST_FOLDER = os.path.join(STORAGE_FOLDER_ROOT, "embeddings")

    os.environ["SENTENCE_TRANSFORMERS_HOME"] = CACHE_FOLDER_ROOT
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_FOLDER_ROOT, "transformers")
    os.environ['HF_HOME'] = os.path.join(CACHE_FOLDER_ROOT,  "hf_home")

    vs = VectorStoreChroma(persist_folder="/storage/llm/embeddings", hf_embed_model_name="all-MiniLM-L6-v2")
    vs.create_index_from_folder(folder_path="/storage/llm/docs", extension="md")
   # retriever = vs.load_retriever()
    
    # query = "How to update a delta table?"
    # r = retriever.get_relevant_documents(query=query)
    # print(r)
