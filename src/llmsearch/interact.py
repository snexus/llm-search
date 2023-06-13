import os
from pathlib import Path

from dotenv import load_dotenv
from termcolor import cprint

from llmsearch.vector_stores import VectorStoreChroma

load_dotenv()

from langchain.chains.question_answering import load_qa_chain

import langchain
langchain.debug = True

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
        print(f"{text[:120]}\n")
        # print(f"\nCONTEXT")
        # print(text)
    print("------------------------------------------")


def qa_with_llm(embedding_persist_folder: str, llm, max_context_size: int, prompt, embedding_model_name: str = "all-MiniLM-L6-v2", chain_type = "stuff"):
    store = VectorStoreChroma(persist_folder=embedding_persist_folder, hf_embed_model_name=embedding_model_name)
    #embed_retriever = store.load_retriever(search_type="similarity", search_kwargs={"k": 10})
    embed_retriever = store.load_retriever(search_type="mmr", search_kwargs={"k": 10})
    chain = load_qa_chain(llm=llm, chain_type=chain_type, prompt = prompt)
    

    while True:
        question = input("\nENTER QUESTION >> ")
        most_relevant_docs = []
        docs = embed_retriever.get_relevant_documents(query=question)
        len_ = 0
        for doc in docs:
            doc_length = len(doc.page_content)
            if len_ + doc_length < max_context_size:
                most_relevant_docs.append(doc)
                len_+= doc_length
        res = chain({"input_documents": most_relevant_docs, "question": question}, return_only_outputs=False)
        print_llm_response(res, prefix=URL_PREFIX)
    
