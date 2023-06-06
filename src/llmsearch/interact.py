import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from termcolor import cprint

from llmsearch.llm import LLMDatabricksDollyV2, LLMMosaicMPT
from llmsearch.vector_stores import VectorStoreChroma

load_dotenv()
from langchain.chains.qa_with_sources import load_qa_with_sources_chain


URL_PREFIX = "obsidian://open?vault=knowledge-base&file="

def print_llm_response(r, prefix: str):
    print("\n============= RESPONSE =================")
    cprint(r["output_text"], "red")
    print("\n============= SOURCES ==================")
    for doc in r["input_documents"]:
        doc_name = Path(doc.metadata["source"]).name
        text = doc.page_content
        cprint(f"{prefix + doc_name}", "blue")

        print("******************* BEING EXTRACT *****************")
        print(f"{text[:200]}\n")
        # print(f"\nCONTEXT")
        # print(text)
    print("------------------------------------------")


def qa_with_llm(embedding_persist_folder: str, llm, k, embedding_model_name: str = "all-MiniLM-L6-v2", chain_type = "stuff"):
    store = VectorStoreChroma(persist_folder=embedding_persist_folder, hf_embed_model_name=embedding_model_name)
    embed_retriever = store.load_retriever(search_type="similarity", search_kwargs={"k": k})
    chain = load_qa_with_sources_chain(llm=llm, chain_type=chain_type)
    while True:
        question = input("\nENTER QUESTION >> ")
        docs = embed_retriever.get_relevant_documents(query=question)
        res = chain({"input_documents": docs, "question": question}, return_only_outputs=False)
        print_llm_response(res, prefix=URL_PREFIX)
    
    

# if __name__ == "__main__":
#     store = VectorStoreChroma(persist_folder=EMBEDDINGS_PERSIST_FOLDER, hf_embed_model_name="all-MiniLM-L6-v2")
#     embed_retriever = store.load_retriever(search_type="similarity", search_kwargs={"k": 3})

#     llm = ChatOpenAI(temperature = 0.0)
#     print("CUDA: ", torch.cuda.is_available())
#     # llm = LLMDatabricksDollyV2(cache_folder=CACHE_FOLDER_ROOT, model_name="databricks/dolly-v2-7b").model
#     #llm = LLMMosaicMPT(cache_folder=CACHE_FOLDER_ROOT, device="cpu").model

#     chain = load_qa_with_sources_chain(llm=llm, chain_type="stuff")

#     while True:
#         question = input("\nENTER QUESTION >> ")
#         docs = embed_retriever.get_relevant_documents(query=question)
#         res = chain({"input_documents": docs, "question": question}, return_only_outputs=False)
#         print_response(res, prefix=URL_PREFIX)
