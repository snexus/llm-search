import os
from pathlib import Path

from dotenv import load_dotenv
from termcolor import cprint

from llmsearch.chroma import VectorStoreChroma
from llmsearch.postprocess import get_and_parse_response
from llmsearch.config import Config, OutputModel


load_dotenv()

from langchain.chains.question_answering import load_qa_chain

import langchain

langchain.debug = True


# def print_llm_response(r, substring_search: str, substring_replace: str):
#     print("\n============= RESPONSE =================")
#     cprint(r["output_text"], "red")
#     print("\n============= SOURCES ==================")
#     for doc in r["input_documents"]:
#         doc_name = doc.metadata["source"]
#         doc_name = doc_name.replace(substring_search, substring_replace)
#         text = doc.page_content
#         cprint(f"{doc_name}", "blue")

#         print("******************* BEING EXTRACT *****************")
#         print(f"{text[:120]}\n")

#     print("------------------------------------------")


def print_llm_response(output: OutputModel):
    print("\n============= RESPONSE =================")
    cprint(output.response, "red")
    print("\n============= SOURCES ==================")
    for source in output.semantic_search:
        cprint(source.chunk_link, "blue")
        print("******************* BEING EXTRACT *****************")
        print(f"{source.chunk_text[:120]}\n")
    print("------------------------------------------")


# def qa_with_llm(embedding_persist_folder: str, llm, max_context_size: int, prompt, substring_search: str, substring_replace: str, chain_type="stuff"):
#     store = VectorStoreChroma(persist_folder=embedding_persist_folder)
#     # embed_retriever = store.load_retriever(search_type="similarity", search_kwargs={"k": 10})
#     embed_retriever = store.load_retriever(search_type="mmr", search_kwargs={"k": 10})
#     chain = load_qa_chain(llm=llm, chain_type=chain_type, prompt=prompt)

#     while True:
#         question = input("\nENTER QUESTION >> ")

#         most_relevant_docs = []
#         docs = embed_retriever.get_relevant_documents(query=question)
#         len_ = 0
#         for doc in docs:
#             doc_length = len(doc.page_content)
#             if len_ + doc_length < max_context_size:
#                 most_relevant_docs.append(doc)
#                 len_ += doc_length
#         res = chain({"input_documents": most_relevant_docs, "question": question}, return_only_outputs=False)
#         print_llm_response(res, substring_search, substring_replace)


def qa_with_llm(llm, prompt: str, config: Config, chain_type="stuff", max_k=10):
    store = VectorStoreChroma(persist_folder=str(config.embeddings.embeddings_path))
    embed_retriever = store.load_retriever(search_type=config.semantic_search.search_type, search_kwargs={"k": max_k})
    chain = load_qa_chain(llm=llm, chain_type=chain_type, prompt=prompt)

    while True:
        question = input("\nENTER QUESTION >> ")
        output = get_and_parse_response(
            prompt=question, chain=chain, embed_retriever=embed_retriever, config=config.semantic_search
        )
        print_llm_response(output)
