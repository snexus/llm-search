# import os

# from llmsearch.chroma import ChromaVS
# from llmsearch.llm import HuggingFaceWrapper, LLMOpenAIWrapper
# from llmsearch.parsers.markdown_v1 import markdown_parser
# from termcolor import colored, cprint

# ## Define fodlers

# STORAGE_FOLDER_ROOT = "/storage/llm/"
# CACHE_FOLDER_ROOT = os.path.join(STORAGE_FOLDER_ROOT, "cache")
# INDEX_PERSIST_FOLDER = os.path.join(STORAGE_FOLDER_ROOT, "index")
# DOC_FOLDER = os.path.join(STORAGE_FOLDER_ROOT, "docs")

# # Define embedding and LLM models
# # EMBEDDING_MODEL = "all-distilroberta-v1" # Assumes HuggingFace embedding.
# EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Assumes HuggingFace embedding.
# URL_PREFIX = "obsidian://open?vault=knowledge-base&file="

# os.environ["SENTENCE_TRANSFORMERS_HOME"] = CACHE_FOLDER_ROOT
# os.environ['TRANSFORMERS_CACHE'] = os.path.join(CACHE_FOLDER_ROOT,  "transformers")
# os.environ['HF_HOME'] = os.path.join(CACHE_FOLDER_ROOT,  "hf_home")


# if __name__ == "__main__":
#     persist_folder = INDEX_PERSIST_FOLDER

#     model = LLMOpenAIWrapper(
#         embedding_model_name=EMBEDDING_MODEL, llm_model_name="gpt-3.5-turbo",
#         temperature=0,
#         max_input_size = 4096,
#         num_output = 2048
#     )

#     # model = HuggingFaceWrapper(
#     #     embedding_model_name=EMBEDDING_MODEL,
#     #     llm_model_name="databricks/dolly-v2-3b",
#     #     cache_folder=CACHE_FOLDER_ROOT,
#     #     max_input_size = 2048,
#     #     num_output = 1024
#     # )

#     vectordb = ChromaVS(
#         persist_folder=persist_folder,
#         collection_name="md_sample",
#         embedding_model_name=EMBEDDING_MODEL,
#         service_context=model.service_context,
#     )

#     # vectordb.create_index_from_folder(folder_path=DOC_FOLDER, extension="md", parser_func=markdown_parser)

#     index = vectordb.load_index()
#     # #  index = vectordb.create_index_from_folder(folder_path=DOC_FOLDER, extension="md", parser_func=markdown_parser, service_context = service_context)

#     # print(index)

#     query_engine = index.as_query_engine(mode="embedding", similarity_top_k=3,
#                                          response_mode="compact",
#                                          verbose=True)
#  r = query_engine.query("How to overwrite a table in Spark?")
# r = query_engine.query("How to convert from string to timestamp in Spark?")
# #r = query_engine.query("What is a difference between transformation and action in Spark?")
# r = query_engine.query("What is buffer pool manager?")

# r = query_engine.query("How to copy and paste in vim, in linux?")

# r = query_engine.query("How to connect to databricks using service principal?")

# r = query_engine.query("Provide minimum working skeleton for pyproject.toml")

# r = query_engine.query("What type of data should be contained in bronze table?")

#    r = query_engine.query("What is a difference between dataclass and regular classes in Python?")

# r = query_engine.query("Provide a code example to update delta table")

# r = query_engine.query("How to check if file exist in bash script?")

# r = query_engine.query("What is a kalman filter?")

# r = query_engine.query("What is a difference between data and non-data descriptor in Python?")

# r = query_engine.query("How to make a class hashable in Python?")

# r = query_engine.query("How data is stored on disk for analytical databases?")

# r = query_engine.query("What type of hashing schemes exist in databases?")

# r = query_engine.query("How to read YAML file into Pydantic model? Provide a code example")

# r = query_engine.query("Provide different methods to update dictionary in Python")

# r = query_engine.query("Provide a code example on how pass parameters to constructor of parent class in Python?")

# #r = query_engine.query("How to use sed to do replacement in strings and write it back to file? Provide an example command.")

# #r = query_engine.query("How to to recursively search files in sub-directories using pathlib in Python?")

# #r = query_engine.query("What is a difference between howto and tutorial in documentation?")
