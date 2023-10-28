import os
from abc import ABC, abstractmethod
from typing import List, Tuple

import pandas as pd

from langchain.embeddings import (
    HuggingFaceEmbeddings,
    HuggingFaceInstructEmbeddings,
    SentenceTransformerEmbeddings,
)
from loguru import logger

from llmsearch.config import Config, Document, EmbeddingModel, EmbeddingModelType
from llmsearch.parsers.splitter import DocumentSplitter
from llmsearch.splade import SparseEmbeddingsSplade

MODELS = {
    EmbeddingModelType.instruct: HuggingFaceInstructEmbeddings,
    EmbeddingModelType.sentence_transformer: SentenceTransformerEmbeddings,
    EmbeddingModelType.huggingface: HuggingFaceEmbeddings,
}


class VectorStore(ABC):
    @abstractmethod
    def create_index_from_documents(self, all_docs: List[Document], clear_persist_folder: bool = True):
        pass

    @abstractmethod
    def get_documents_by_id(self, docuemt_ids: List[str]):
        pass

    @property
    @abstractmethod
    def retriever(self):
        pass

    @abstractmethod
    def similarity_search_with_relevance_scores(self, query: str, k: int, filter: dict):
        pass


def get_embedding_model(config: EmbeddingModel):
    """Loads an embedidng model

    Args:
        config (EmbeddingModel): Configuration for the embedding model

    Raises:
        TypeError: if model is unsupported
    """

    logger.info(f"Embedding model config: {config}")
    model_type = MODELS.get(config.type, None)

    if model_type is None:
        raise TypeError(f"Unknown model type. Got {config.type}")

    return model_type(model_name=config.model_name, **config.additional_kwargs)


def create_embeddings(config: Config, vs: VectorStore):
    splitter = DocumentSplitter(config)
    all_docs, all_hash_filename_mappings, all_hash_docid_mappings = splitter.split()

    vs.create_index_from_documents(all_docs=all_docs)

    splade = SparseEmbeddingsSplade(config)
    splade.generate_embeddings_from_docs(docs=all_docs)

    save_document_hashes(config, all_hash_filename_mappings, all_hash_docid_mappings)
    logger.info("ALL DONE.")


def update_embeddings(config: Config, vs: VectorStore):
    splitter = DocumentSplitter(config)
    new_hashes_df = splitter.get_hashes()
    file_hashes_fn, _ = get_hash_mapping_filenames(config)
    existing_hashes_df = pd.read_parquet(path=file_hashes_fn)

    print(new_hashes_df)

    changed_or_new_df = get_changed_or_new_files(new_hashes_df, existing_hashes_df)
    print("CHanged or new files")
    print(changed_or_new_df)


def get_changed_or_new_files(new_hashes_df: pd.DataFrame, existing_hashes_df: pd.DataFrame) -> pd.DataFrame:
    changed_or_new = (
        new_hashes_df.merge(existing_hashes_df, on=["filehash", "filename"], how="outer", indicator=True)
        #   .loc[lambda df: df['_merge'] == 'left_only']
    )
    return changed_or_new


def get_hash_mapping_filenames(
    config: Config,
    file_to_hash_fn: str = "file_hash_mappings.snappy.parquet",
    docid_to_hash_fn="docid_hash_mappings.snappy.parquet",
) -> Tuple[str, str]:
    """Returns filenames to store hashes"""

    file_hashes_fn = os.path.join(config.embeddings.embeddings_path, file_to_hash_fn)
    docid_hashes_fn = os.path.join(config.embeddings.embeddings_path, docid_to_hash_fn)
    return file_hashes_fn, docid_hashes_fn


def save_document_hashes(
    config: Config,
    all_hash_filename_mappings: pd.DataFrame,
    all_hash_docid_mappings: pd.DataFrame,
) -> None:
    """Saves hashes of the processed files and mappings from hashes to individual chunks

    Args:
        config (Config): Instance of Config
        all_hash_filename_mappings (pd.DataFrame): Dataframe containing mappings between filename and file hash
        all_hash_docid_mappings (pd.DataFrame): Dataframe contains mappings between file hash and document ids
    """

    file_hashes_fn, docid_hashes_fn = get_hash_mapping_filenames(config)

    logger.info(f"Saving file hashes mappings to {file_hashes_fn}")
    all_hash_filename_mappings.to_parquet(file_hashes_fn, compression="snappy", index=False)
    logger.info(f"Distinct hashes generated: {len(all_hash_filename_mappings['filehash'].unique())}")

    logger.info(f"Saving document id to hash mappings to {docid_hashes_fn}")
    all_hash_docid_mappings.to_parquet(docid_hashes_fn, compression="snappy", index=False)
    logger.info(f"Hash to document id mappings: {len(all_hash_docid_mappings)}")
