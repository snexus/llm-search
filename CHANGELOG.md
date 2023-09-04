## 2023-09-03

* Add support for labels and ability to filter the results by labels.
* Bump versions of the dependencies

## 2023-08-28

* Added support for saving LLM responses with corresponding sources to offline database
* Added support configuring batch size for generation of SPLADE embeddings. Useful for low-memory GPUs.

## 2023-08-22

* Implement hybrid search (sparse + dense embeddings). Sparse embeddings are implemented using SPLADE. Hybrid search is enabled by default.

## 2023-08-18

* Ability to split documents by multiple chunk sizes at once (supported by chunk_size parameter in config.yaml). The change is a breaking change for the configuration, check the updated templates. During run time, best chunk size is selected based on aggregated score from re-ranker.


## 2023-08-12

* Abiility to add prefixes for embedded documents the query. Prefixes are often required for the embedding models for asymmetric queries (when short query is being matched to a long text paragraph) - see for example https://huggingface.co/intfloat/e5-large-v2#faq


## 2023-08-11

* Added an ability to re-rank documents after retrieving from vector database, using cross-encoder - models - see https://www.sbert.net/examples/applications/retrieve_rerank/README.html
    * This behaviour is controlled be `reranker: True` parameter in semantic_search section of configuration
* Added an ability to specify maximum number of retrieved documents using `k_max` paramters in semantic_search section
* Refactoring and cleaning up the code.



## 2023-07-30

* Code cleaning and refactoring

* Improvements to the markdown parser:
    - Added options to clean markdown before processing, which includes removing image links and extra new lines.
    - Implemented the ability to extract custom metadata and attach it to every output text chunk.

* Enhancements to document management:
    - Now supports including multiple document paths (refer to the new format of config.yaml for details).
    - Added the ability to perform multiple search/replace substitutions for the output paths.

* Experimental web interface (Streamlit):
