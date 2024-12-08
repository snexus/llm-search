import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple, Union

from dotenv import load_dotenv
import pandas as pd
from langchain_community.embeddings import (
    HuggingFaceInstructEmbeddings,
    SentenceTransformerEmbeddings,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from loguru import logger

from llmsearch.config import Config, Document, EmbeddingModel, EmbeddingModelType
from llmsearch.parsers.splitter import DocumentSplitter
from llmsearch.splade import SparseEmbeddingsSplade

load_dotenv()

MODELS = {
    EmbeddingModelType.instruct: HuggingFaceInstructEmbeddings,
    EmbeddingModelType.sentence_transformer: SentenceTransformerEmbeddings,
    EmbeddingModelType.huggingface: HuggingFaceEmbeddings,
}


class VectorStore(ABC):
    @abstractmethod
    def create_index_from_documents(
        self, all_docs: List[Document], clear_persist_folder: bool = True
    ):
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

    @abstractmethod
    def delete_by_id(self, ids: List[str]):
        pass

    @abstractmethod
    def add_documents(self, docs: List[Document]):
        pass


class EmbeddingsHashNotExistError(Exception):
    """Raised when embeddings can't be founrd"""


def get_embedding_model(config: EmbeddingModel):
    """Loads an embedidng model

    Args:
        config (EmbeddingModel): Configuration for the embedding model

    Raises:
        TypeError: if model is unsupported
    """

    logger.info(f"Embedding model config: {config}")

    if config.type in (
        EmbeddingModelType.huggingface,
        EmbeddingModelType.instruct,
        EmbeddingModelType.sentence_transformer,
    ):
        model_type: Optional[
            Union[
                HuggingFaceEmbeddings,
                HuggingFaceInstructEmbeddings,
                SentenceTransformerEmbeddings,
            ]
        ] = MODELS.get(config.type, None) # type: ignore

        if model_type is None:
            raise TypeError(f"Invalid model type passed: {config.type}")
        return model_type(
            model_name=config.model_name, model_kwargs=config.additional_kwargs
        )  # type: ignore

    elif config.type is EmbeddingModelType.openai:
        return get_openai_embedding_model(config)
    else:
        raise TypeError(f"Unknown model type. Got {config.type}")
    # return model_type(model_name=config.model_name, **config.additional_kwargs)


def get_openai_embedding_model(config: EmbeddingModel) -> OpenAIEmbeddings:
    if not os.getenv("OPENAI_API_KEY"):
        raise KeyError("OPENAI_API_KEY wasn't found. Please refer to .env_template to create .env")

    logger.info("Initializing OpenAI embeddings model.")
    return OpenAIEmbeddings(
        model = config.model_name,
        **config.additional_kwargs
    )



def create_embeddings(config: Config, vs: VectorStore):
    splitter = DocumentSplitter(config)
    all_docs, all_hash_filename_mappings, all_hash_docid_mappings, all_labels = (
        splitter.split()
    )

    vs.create_index_from_documents(all_docs=all_docs)

    splade = SparseEmbeddingsSplade(config)
    splade.generate_embeddings_from_docs(docs=all_docs)

    save_document_hashes(config, all_hash_filename_mappings, all_hash_docid_mappings)
    update_document_labels(config, all_labels)
    logger.info("ALL DONE.")


def update_document_labels(config: Config, all_labels: List[str]):
    logger.info("Updating document labels...")
    labels_fn = Path(os.path.join(config.embeddings.embeddings_path, "labels.txt"))
    labels = load_document_labels(labels_fn)

    labels = list(set(labels + all_labels))
    save_document_labels(labels_fn, labels)


def load_document_labels(path: Path) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            strings = [line.strip() for line in f.readlines()]
            return strings
    except FileNotFoundError:
        logger.warning("List of labels wasn't found, returning []")
        return []


def save_document_labels(path: Path, labels: List[str]) -> None:
    logger.info(f"Saving document labels to {path}")
    with open(path, "w", encoding="utf-8") as f:
        for string in labels:
            f.write(string + "\n")  # Write each string followed by a newline


def update_embeddings(config: Config, vs: VectorStore) -> dict:
    splitter = DocumentSplitter(config)
    new_hashes_df = splitter.get_hashes()
    try:
        file_hashes_fn, docid_hash_fn = get_hash_mapping_filenames(config)
        existing_fn_hash_mappings = pd.read_parquet(path=file_hashes_fn)
    except FileNotFoundError:
        raise EmbeddingsHashNotExistError(
            "Hash file don't exist, please re-create the index."
        )
    existing_docid_hash_mappings = pd.read_parquet(path=docid_hash_fn)
    stats = {
        "original_n_files": len(existing_fn_hash_mappings),
        "updated_n_files": 0,
        "scanned_files": 0,
        "scanned_chunks": 0,
        "changed_files": 0,
        "changed_chunks": 0,
        "deleted_files": 0,
        "deleted_chunks": 0,
    }

    changed_or_new_df, changed_df, deleted_df = get_changed_or_new_files(
        new_hashes_df, existing_fn_hash_mappings
    )
    logger.debug("===== Rescanned files ======")
    logger.debug(changed_or_new_df)
    logger.debug("===== Changed files ======")
    logger.debug(changed_df)
    logger.debug("===== Deleted files ======")
    logger.debug(deleted_df)

    if len(changed_or_new_df) == 0 and len(changed_df) == 0 and len(deleted_df) == 0:
        logger.info("The index is up to date. Exiting.")
        return stats

    splade = SparseEmbeddingsSplade(config=config)
    splade.load()

    # Delete chunks belonging to changed documents that exist in both new and old index
    if len(changed_df) > 0:
        # print("Changed filename:")
        # print(changed_df['filename'].tolist())
        changed_doc_ids = existing_docid_hash_mappings.loc[
            existing_docid_hash_mappings["filehash"].isin(changed_df["filehash"]),
            "docid",
        ].tolist()

        logger.info(
            "Removing chunks from vectorstore belonging to changed documents..."
        )
        # print("CHANGED ids", changed_doc_ids)
        if changed_doc_ids:
            vs.delete_by_id(ids=changed_doc_ids)
        logger.info("Removing mappings belonging to changed documents...")
        existing_fn_hash_mappings, existing_docid_hash_mappings = delete_mappings(
            existing_fn_hash_mappings, existing_docid_hash_mappings, changed_df
        )

        # Delete splade embeddings
        if changed_doc_ids:
            splade.delete_by_ids(delete_ids=changed_doc_ids)
        stats["changed_files"] = len(changed_df)
        stats["changed_chunks"] = len(changed_doc_ids)

    # Delete chunks belonging to deleted documents that exist in only in old index
    if len(deleted_df) > 0:
        logger.info(
            "Removing chunks from vectorstore belonging to deleted documents..."
        )
        deleted_doc_ids = existing_docid_hash_mappings.loc[
            existing_docid_hash_mappings["filehash"].isin(deleted_df["filehash"]),
            "docid",
        ].tolist()

        if deleted_doc_ids:
            vs.delete_by_id(ids=deleted_doc_ids)

        logger.info("Removing mappings belonging to deleted documents...")
        existing_fn_hash_mappings, existing_docid_hash_mappings = delete_mappings(
            existing_fn_hash_mappings, existing_docid_hash_mappings, deleted_df
        )

        # Delete splade embeddings
        if deleted_doc_ids:
            splade.delete_by_ids(delete_ids=deleted_doc_ids)
        stats["deleted_files"] = len(deleted_df)
        stats["deleted_chunks"] = len(deleted_doc_ids)

    # Rescan new and changed documents and add them to vectorstore
    if len(changed_or_new_df) > 0:
        splitter = DocumentSplitter(config)

        new_docs, new_fn_hash_mappings, new_docid_hash_mappings, all_labels = (
            splitter.split(
                restrict_filenames=changed_or_new_df.loc[:, "filename"].tolist()
            )
        )

        existing_fn_hash_mappings, existing_docid_hash_mappings = add_mappings(
            existing_fn_hash_mappings,
            new_fn_hash_mappings,
            existing_docid_hash_mappings,
            new_docid_hash_mappings,
        )
        vs.add_documents(new_docs)

        splade.add_embeddings(new_docs)
        stats["scanned_files"] = len(changed_or_new_df)
        stats["scanned_chunks"] = len(new_docs)
        update_document_labels(config, all_labels)

    stats["updated_n_files"] = len(existing_fn_hash_mappings)
    # Save changed mappings
    save_document_hashes(
        config, existing_fn_hash_mappings, existing_docid_hash_mappings
    )
    return stats


def delete_mappings(
    existing_fn_hash_mappings: pd.DataFrame,
    existing_docid_hash_mappings: pd.DataFrame,
    changed_df,
):
    # In the existing mappings, delete all rows belonging to changed filenames
    mask_delete_fn_hash = existing_fn_hash_mappings.loc[:, "filename"].isin(
        changed_df.loc[:, "filename"]
    )
    updated_fn_hash_mappings = existing_fn_hash_mappings.loc[~mask_delete_fn_hash, :]

    # Update existings mappings with new fn mappings
    mask_delete_docid_hash = existing_docid_hash_mappings.loc[:, "filehash"].isin(
        changed_df.loc[:, "filehash"]
    )
    updated_docid_hash_mappings = existing_docid_hash_mappings.loc[
        ~mask_delete_docid_hash, :
    ]
    return updated_fn_hash_mappings, updated_docid_hash_mappings


def add_mappings(
    existing_fn_hash_mappings: pd.DataFrame,
    new_fn_hash_mappings: pd.DataFrame,
    existing_docid_hash_mappings: pd.DataFrame,
    new_docid_hash_mappings: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Adds new hash and docid mappings to the existing ones.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: updated filename to hash mapping df,
            updated docid to hash mapping df
    """

    # Update existings mappings with new fn mappings
    updated_fn_hash_mappings = pd.concat(
        [existing_fn_hash_mappings, new_fn_hash_mappings], axis=0
    )

    updated_docid_hash_mappings = pd.concat(
        [existing_docid_hash_mappings, new_docid_hash_mappings], axis=0
    )
    return updated_fn_hash_mappings, updated_docid_hash_mappings


def get_changed_or_new_files(
    new_hashes_df: pd.DataFrame, existing_hashes_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculates changes between new file hashes and existing file hashes

    Args:
        new_hashes_df (pd.DataFrame): Dataframe containing newly scanned file hashes
        existing_hashes_df (pd.DataFrame): Dataframe containing existing file hashes


    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Dataframes containing:
            changed or new files - these files will need to be (re)scanned.
            changed_files - these files to be rescanned, and old docids to be deleted from the vector store
            deleted - docids belonging to these files have to be deleted from the vector store.
    """

    merged_df = new_hashes_df.merge(
        existing_hashes_df, on=["filehash", "filename"], how="outer", indicator=True
    )

    # Rows with left_only indicators contain files that are either changed or new in the new scan
    # These files are to be re-scanned
    changed_or_new = merged_df.loc[lambda df: df["_merge"] == "left_only"]

    duplicated_filenames_mask = merged_df.loc[:, "filename"].duplicated()
    changed = merged_df.loc[duplicated_filenames_mask, :]

    # Deleted files - mask is "right_only" (existed only in the previous df), but also not in the changed files
    deleted_mask = (merged_df["_merge"] == "right_only") & (
        ~merged_df["filename"].isin(changed["filename"])
    )
    deleted = merged_df.loc[deleted_mask, :]

    return changed_or_new, changed, deleted


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
    all_hash_filename_mappings.to_parquet(
        file_hashes_fn, compression="snappy", index=False
    )
    logger.info(
        f"Distinct hashes generated: {len(all_hash_filename_mappings['filehash'].unique())}"
    )

    logger.info(f"Saving document id to hash mappings to {docid_hashes_fn}")
    all_hash_docid_mappings.to_parquet(
        docid_hashes_fn, compression="snappy", index=False
    )
    logger.info(f"Hash to document id mappings: {len(all_hash_docid_mappings)}")
