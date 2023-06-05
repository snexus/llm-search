import os
from pathlib import Path

from dotenv import load_dotenv
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from termcolor import cprint
import torch

from llmsearch.vector_stores import VectorStoreChroma
from llmsearch.llm import LLMDatabricksDollyV2, LLMMosaicMPT

load_dotenv()
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain

STORAGE_FOLDER_ROOT = "/storage/llm/"
CACHE_FOLDER_ROOT = os.path.join(STORAGE_FOLDER_ROOT, "cache")
EMBEDDINGS_PERSIST_FOLDER = os.path.join(STORAGE_FOLDER_ROOT, "embeddings")
URL_PREFIX = "obsidian://open?vault=knowledge-base&file="

os.environ["SENTENCE_TRANSFORMERS_HOME"] = CACHE_FOLDER_ROOT
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_FOLDER_ROOT, "transformers")
os.environ["HF_HOME"] = os.path.join(CACHE_FOLDER_ROOT, "hf_home")


def print_response(r, prefix: str):
    print("\n============= RESPONSE =================")
    cprint(res["output_text"], "red")
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


if __name__ == "__main__":
    store = VectorStoreChroma(persist_folder=EMBEDDINGS_PERSIST_FOLDER, hf_embed_model_name="all-MiniLM-L6-v2")
    embed_retriever = store.load_retriever(search_type="similarity", search_kwargs={"k": 3})

    # llm = ChatOpenAI(temperature = 0.0)
    print("CUDA: ", torch.cuda.is_available())
    llm = LLMDatabricksDollyV2(cache_folder=CACHE_FOLDER_ROOT, model_name="databricks/dolly-v2-7b").model
    #llm = LLMMosaicMPT(cache_folder=CACHE_FOLDER_ROOT, device="cpu").model

    chain = load_qa_with_sources_chain(llm=llm, chain_type="stuff")

    while True:
        question = input("\nENTER QUESTION >> ")
        docs = embed_retriever.get_relevant_documents(query=question)
        res = chain({"input_documents": docs, "question": question}, return_only_outputs=False)
        print_response(res, prefix=URL_PREFIX)
