import os
import pickle
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import scipy
import torch
import tqdm
from loguru import logger
from transformers import AutoModelForMaskedLM, AutoTokenizer

# from llmsearch.utils import set_cache_folder
from llmsearch.config import Config, Document


def split(iterable: List, chunk_size: int):
    """Splits a list to chunks of size `chunk_size`"""

    for i in range(0, len(iterable), chunk_size):
        yield iterable[i : i + chunk_size]


class SparseEmbeddingsSplade:
    def __init__(
        self,
        config: Config,
        splade_model_id: str = "naver/splade-cocondenser-ensembledistil",
    ) -> None:
        self._config = config

        self._device = (
            f"cuda:{torch.cuda.current_device()}"
            if torch.cuda.is_available()
            else "cpu"
        )
        logger.info(f"Setting device to {self._device}")

        #        set_cache_folder(str(config.cache_folder))
        self.tokenizer = AutoTokenizer.from_pretrained(
            splade_model_id, device=self._device, use_fast=True
        )
        self.model = AutoModelForMaskedLM.from_pretrained(splade_model_id)
        self.model.to(self._device)
        self._embeddings = None
        self._ids = None
        self._l2_norm_matrix = None
        self._labels_to_ind = defaultdict(list)
        self._chunk_size_to_ind = defaultdict(list)

        self.n_batch = config.embeddings.splade_config.n_batch

    def _get_batch_embeddings(
        self, docs: List[str], free_memory: bool = True
    ) -> np.ndarray:
        tokens = self.tokenizer(
            docs, return_tensors="pt", padding=True, truncation=True
        ).to(self._device)

        output = self.model(**tokens)

        # aggregate the token-level vecs and transform to sparse
        vecs = (
            torch.max(
                torch.log(1 + torch.relu(output.logits))
                * tokens.attention_mask.unsqueeze(-1),
                dim=1,
            )[0]
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        # For smaller VRAM sizes, might be useful to free the memory.
        if free_memory:
            del output
            del tokens
            torch.cuda.synchronize()

        return vecs

    def _get_embedding_fnames(self):
        folder_name = os.path.join(self._config.embeddings.embeddings_path, "splade")
        fn_embeddings = os.path.join(folder_name, "splade_embeddings.npz")
        fn_ids = os.path.join(folder_name, "splade_ids.pickle")
        fn_metadatas = os.path.join(folder_name, "splade_metadatas.pickle")
        return folder_name, fn_embeddings, fn_ids, fn_metadatas

    def load(self) -> None:
        """Loads embeddings from disk."""

        _, fn_embeddings, fn_ids, fn_metadatas = self._get_embedding_fnames()
        try:
            self._embeddings = scipy.sparse.load_npz(fn_embeddings)
            with open(fn_ids, "rb") as fp:
                self._ids = np.array(pickle.load(fp))

            # Load metadata
            with open(fn_metadatas, "rb") as fm:
                self._metadatas = np.array(pickle.load(fm))

            self._l2_norm_matrix = scipy.sparse.linalg.norm(self._embeddings, axis=1)

            for ind, m in enumerate(self._metadatas):
                if m["label"]:
                    self._labels_to_ind[m["label"]].append(ind)

                self._chunk_size_to_ind[m["chunk_size"]].append(ind)

            logger.info(f"SPLADE: Got {len(self._labels_to_ind)} labels.")

        except FileNotFoundError:
            raise FileNotFoundError(
                "Embeddings don't exist, run generate_embeddings_from_docs(..) first."
            )
        logger.info(f"Loaded sparse (SPLADE) embeddings from {fn_embeddings}")

    def generate_embeddings_from_docs(
        self, docs: List[Document], persist: bool = True
    ) -> Tuple[np.ndarray, List[str], List[dict]]:
        """Generates SPLADE embeddings from documents, in batches of chunk_size.

        Args:
            docs (List[Document]): list of Documents to generate the embeddings for.
            chunk_size (int, optional): Number of documents to process per batch.
                                        Defaults to 5. Can be higher for larger VRAM.
        """

        chunk_size = self.n_batch
        logger.info(
            f"Calculating SPLADE embeddings for {len(docs)} documents. Using chunk size: {chunk_size}"
        )

        ids = [d.metadata["document_id"] for d in docs]
        metadatas = [d.metadata for d in docs]

        vecs = []
        for chunk in tqdm.tqdm(
            split(docs, chunk_size=chunk_size), total=int(len(docs) / chunk_size)
        ):
            texts = [d.page_content for d in chunk if d.page_content]
            vecs.append(self._get_batch_embeddings(texts))

        embeddings = np.vstack(vecs)
        logger.info(f"Shape of the embeddings matrix: {embeddings.shape}")

        if persist:
            self.persist_embeddings(embeddings, metadatas, ids)
        return embeddings, ids, metadatas

    def add_embeddings(self, new_docs: List[Document]) -> None:
        if self._embeddings is None or self._ids is None:
            logger.info("Loading embeddings...")
            self.load()

        embeddings, ids, metadatas = self.generate_embeddings_from_docs(
            docs=new_docs, persist=False
        )
        if self._ids is not None and self._embeddings is not None:
            logger.debug(
                f"Splade embeddings shape before update: {self._embeddings.shape}"
            )
            updated_embeddings = np.vstack((self._embeddings.toarray(), embeddings))
            logger.debug(
                f"Splade embeddings shape after update: {updated_embeddings.shape}"
            )
            updated_ids = self._ids.tolist() + ids
            updated_metadata = self._metadatas.tolist() + metadatas
            logger.debug(f"Splade: Updated metadata length: {len(updated_metadata)}")
            logger.debug(f"Splade: Updated ids length: {len(updated_ids)}")
            self.persist_embeddings(updated_embeddings, updated_metadata, updated_ids)
        else:
            raise Exception(
                "Something is wrong: ids and embeddings weren't loaded properly."
            )

    def persist_embeddings(self, embeddings, metadatas, ids):
        folder_name, fn_embeddings, fn_ids, fn_metadatas = self._get_embedding_fnames()
        csr_embeddings = scipy.sparse.csr_matrix(embeddings)

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        scipy.sparse.save_npz(fn_embeddings, csr_embeddings)
        self.save_list(ids, fn_ids)
        self.save_list(metadatas, fn_metadatas)
        logger.info(f"Saved SPLADE embeddings to {fn_embeddings}")

    def delete_by_ids(self, delete_ids: List[str]):
        if self._embeddings is None or self._ids is None:
            logger.info("Loading embeddings...")
            self.load()

        if self._ids is not None and self._embeddings is not None:
            indices = [
                ind for ind, id_ in enumerate(self._ids) if id_ not in set(delete_ids)
            ]
            self._ids = self._ids[indices]
            self._embeddings = self._embeddings[indices]
            self._metadatas = self._metadatas[indices]

        else:
            raise Exception(
                "Something is wrong: ids and embeddings weren't loaded properly."
            )

    def query(
        self, search: str, chunk_size: int, n: int = 50, label: str = ""
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Finds documents using sparse index similar to the search query in meaning

        Args:
            search (str): Search query
            n (int, optional): Number of document ids to return. Defaults to 50.

        Raises:
            Exception: If no persitent embeedings were found

        Returns:
            List[str], List[float]: list of document ids with scores, most similar to the search query
        """

        if self._embeddings is None or self._ids is None:
            logger.info("Loading embeddings...")
            self.load()

        # If label is present, restrict the search to a subset of documents containing the label
        if (
            label
            and label in self._labels_to_ind
            and self._embeddings is not None
            and self._ids is not None
        ):
            indices = sorted(
                list(
                    set(self._labels_to_ind[label]).intersection(
                        set(self._chunk_size_to_ind[chunk_size])
                    )
                )
            )
            logger.info(
                f"SPLADE - restricting search to label: {label}, chunk size: {chunk_size}. Number of docs: {len(indices)}"
            )

        else:
            indices = sorted(list(set(self._chunk_size_to_ind[chunk_size])))
            logger.info(
                f"SPLADE search will search over all documents of chunk size: {chunk_size}. Number of docs: {len(indices)}"
            )

        # print(indices)
        embeddings = self._embeddings[indices]  # type: ignore
        ids = self._ids[indices]  # type: ignore
        l2_norm_matrix = scipy.sparse.linalg.norm(embeddings, axis=1)

        embed_query = self._get_batch_embeddings(docs=[search])
        l2_norm_query = scipy.linalg.norm(embed_query)

        if embeddings is not None and l2_norm_matrix is not None and ids is not None:
            cosine_similarity = embeddings.dot(embed_query) / (
                l2_norm_matrix * l2_norm_query
            )
            # print(cosine_similarity)
            most_similar = np.argsort(cosine_similarity)

            top_similar_indices = most_similar[-n:][::-1]
            return (
                ids[top_similar_indices],
                cosine_similarity[top_similar_indices],
            )
        else:
            raise Exception(
                "Something went wrong..Embeddings weren't calculated or loaded properly."
            )

    def save_list(self, list_: list, fname: str) -> None:
        # store list in binary file so 'wb' mode
        with open(fname, "wb") as fp:
            pickle.dump(list_, fp)
