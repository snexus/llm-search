from llmplay.chroma import ChromaVS
from llmplay.llm import LLMOpenAI
from llmplay.parsers import markdown_parser

# DOC_FOLDER = "/shared/sample_data/markdown"
DOC_FOLDER = "/shared/temp_data/"
EMBEDDING_MODEL = "all-distilroberta-v1"

import os
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/shared/sample_data/cache"

if __name__ == "__main__":
    persist_folder = "/shared/sample_data/md_index2"

    model = LLMOpenAI(
        embedding_model_name=EMBEDDING_MODEL, llm_model_name="gpt-3.5-turbo"
    )

    vectordb = ChromaVS(
        persist_folder=persist_folder,
        collection_name="md_sample",
        embedding_model_name=EMBEDDING_MODEL,
        service_context=model.service_context,
    )

  #  vectordb.create_index_from_folder(folder_path=DOC_FOLDER, extension="md", parser_func=markdown_parser)

    index = vectordb.load_index()
    # #  index = vectordb.create_index_from_folder(folder_path=DOC_FOLDER, extension="md", parser_func=markdown_parser, service_context = service_context)

    # print(index)

    query_engine = index.as_query_engine(mode="embedding", similarity_top_k=3, 
                                         response_mode="compact",
                                         verbose=True)
   #  r = query_engine.query("How to overwrite a table in Spark?")
   # r = query_engine.query("How to convert from string to timestamp in Spark?")
    # #r = query_engine.query("What is a difference between transformation and action in Spark?")
    #r = query_engine.query("What is buffer pool manager?")
    
    # r = query_engine.query("How to copy and paste in vim, in linux?")
    
    #r = query_engine.query("How to connect to databricks using service principal?")
    
    # r = query_engine.query("Provide minimum working skeleton for pyproject.toml")
    
    # r = query_engine.query("What type of data should be container in bronze table?")
    
#    r = query_engine.query("What is a difference between dataclass and regular classes in Python?")
    
    # r = query_engine.query("Provide a code example to update delta table")
    
    # r = query_engine.query("How to check if file exist in bash script?")
    
    #r = query_engine.query("What is a kalman filter?")
    
    # r = query_engine.query("What is a difference between data and non-data descriptor in Python?")
    
    # r = query_engine.query("How to make a class hashable in Python?")
    
    #r = query_engine.query("How data is stored on disk for analytical databases?")
    
    # r = query_engine.query("What type of hashing schemes exist in databases?")
    
    # r = query_engine.query("How to read YAML file into Pydantic model? Provide a code example")
    
    # r = query_engine.query("Provide different methods to update dictionary in Python")
    
    # r = query_engine.query("Provide a code example on how pass parameters to constructor of parent class in Python?")
    
    #r = query_engine.query("How to use sed to do replacement in strings and write it back to file? Provide an example command.")
    
    r = query_engine.query("How to to recursively search files in sub-directories using pathlib in Python?")
    
    print("\n============= RESPONSE =================")
    print(r.response)
    print("\n============= SOURCES ==================")
    for node in r.source_nodes:
        doc_name = node.node.doc_id[3:-3]
        score = node.score
        text = node.node.text
        print(f"Document name: {doc_name}, score: {score:.3f}")
        # print(f"\nCONTEXT")
        # print(text)
    print('------------------------------------------\n\n')
        
        
        

        
        
        
    
    
    # print(r)
