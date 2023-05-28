from llmplay.chroma import ChromaVS
from llmplay.llm import LLMOpenAI


DOC_FOLDER = "/shared/sample_data/markdown"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

if __name__ == "__main__":
    persist_folder = "/shared/sample_data/md_index"

    model = LLMOpenAI(
        embedding_model_name=EMBEDDING_MODEL, llm_model_name="gpt-3.5-turbo", cache_fodler="/shared/sample_data/cache"
    )

    vectordb = ChromaVS(
        persist_folder=persist_folder,
        collection_name="md_sample",
        embedding_model_name=EMBEDDING_MODEL,
        service_context=model.service_context,
    )

    # vectordb.create_index_from_folder(folder_path=DOC_FOLDER, extension="md", parser_func=markdown_parser)

    index = vectordb.load_index()
    #  index = vectordb.create_index_from_folder(folder_path=DOC_FOLDER, extension="md", parser_func=markdown_parser, service_context = service_context)

    print(index)

    query_engine = index.as_query_engine(mode="embedding", similarity_top_k=3, response_mode="compact", verbose=True)
    r = query_engine.query("How to overwrite a table in Spark?")
    #r = query_engine.query("What is a difference between transformation and action in Spark?")
    print(r)
