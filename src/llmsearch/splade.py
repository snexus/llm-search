import os
import pickle
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
    def __init__(self, config: Config, splade_model_id: str = "naver/splade-cocondenser-ensembledistil") -> None:
        self._config = config

        self._device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        logger.info(f"Setting device to {self._device}")

        #        set_cache_folder(str(config.cache_folder))
        self.tokenizer = AutoTokenizer.from_pretrained(splade_model_id, device=self._device, use_fast=True)
        self.model = AutoModelForMaskedLM.from_pretrained(splade_model_id)
        self.model.to(self._device)
        self._embeddings = None
        self._ids = None
        self._l2_norm_matrix = None

    def _get_batch_embeddings(self, docs: List[str], free_memory: bool = True) -> np.ndarray:
        tokens = self.tokenizer(docs, return_tensors="pt", padding=True, truncation=True).to(self._device)

        output = self.model(**tokens)

        # aggregate the token-level vecs and transform to sparse
        vecs = (
            torch.max(torch.log(1 + torch.relu(output.logits)) * tokens.attention_mask.unsqueeze(-1), dim=1)[0]
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
        return folder_name, fn_embeddings, fn_ids

    def load(self) -> None:
        """Loads embeddings from disk."""

        _, fn_embeddings, fn_ids = self._get_embedding_fnames()
        try:
            self._embeddings = scipy.sparse.load_npz(fn_embeddings)
            with open(fn_ids, "rb") as fp:
                self._ids = np.array(pickle.load(fp))
            self._l2_norm_matrix = scipy.sparse.linalg.norm(self._embeddings, axis=1)
        except FileNotFoundError:
            raise FileNotFoundError("Embeddings don't exist, run generate_embeddings_from_docs(..) first.")
        logger.info(f"Loaded sparse (SPLADE) embeddings from {fn_embeddings}")

    def generate_embeddings_from_docs(self, docs: List[Document], chunk_size: int = 5, persist: bool = True):
        """Generates SPLADE embeddings from documents, in batches of chunk_size.

        Args:
            docs (List[Document]): list of Documents to generate the embeddings for.
            chunk_size (int, optional): Number of documents to process per batch. Defaults to 5. Can be higher for larger VRAM.
        """

        logger.info(f"Calculating SPLADE embeddings for {len(docs)} documents.")

        ids = [d.metadata["document_id"] for d in docs]

        vecs = []
        for chunk in tqdm.tqdm(split(docs, chunk_size=chunk_size)):
            texts = [d.page_content for d in chunk]
            vecs.append(self._get_batch_embeddings(texts))

        vecs_flat = [item for row in vecs for item in row]
        embeddings = np.stack(vecs_flat)

        csr_embeddings = scipy.sparse.csr_matrix(embeddings)

        if persist:
            folder_name, fn_embeddings, fn_ids = self._get_embedding_fnames()

            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            scipy.sparse.save_npz(fn_embeddings, csr_embeddings)
            self.save_list(ids, fn_ids)
            logger.info(f"Saved embeddings to {fn_embeddings}")
        return csr_embeddings, ids

    def query(self, search: str, n: int = 50) -> Tuple[List[str], np.array]:
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

        embed_query = self._get_batch_embeddings(docs=[search])
        l2_norm_query = scipy.linalg.norm(embed_query)

        if self._embeddings is not None and self._l2_norm_matrix is not None and self._ids is not None:
            cosine_similarity = self._embeddings.dot(embed_query) / (self._l2_norm_matrix * l2_norm_query)
            print(cosine_similarity)
            most_similar = np.argsort(cosine_similarity)

            top_similar_indices = most_similar[-n:][::-1]
            return self._ids[top_similar_indices], cosine_similarity[top_similar_indices]
        else:
            raise Exception("Something went wrong..Embeddings weren't calculated or loaded properly.")

    def save_list(self, list_: list, fname: str) -> None:
        # store list in binary file so 'wb' mode
        with open(fname, "wb") as fp:
            pickle.dump(list_, fp)
