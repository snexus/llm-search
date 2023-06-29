import os
from pathlib import Path

from dotenv import load_dotenv
from termcolor import cprint

from llmsearch.chroma import VectorStoreChroma
from llmsearch.process import get_and_parse_response
from llmsearch.config import Config, OutputModel


load_dotenv()

from langchain.chains.question_answering import load_qa_chain

import langchain

langchain.debug = True


def print_llm_response(output: OutputModel):
    print("\n============= RESPONSE =================")
    cprint(output.response, "red")
    print("\n============= SOURCES ==================")
    for source in output.semantic_search:
        source.metadata.pop('source')
        cprint(source.chunk_link, "blue")
        cprint(source.metadata, "cyan")
        print("******************* BEING EXTRACT *****************")
        print(f"{source.chunk_text}\n")
    print("------------------------------------------")


def qa_with_llm(llm, prompt: str, config: Config, chain_type="stuff", max_k=7):
    store = VectorStoreChroma(persist_folder=str(config.embeddings.embeddings_path))
    embed_retriever = store.load_retriever(search_type=config.semantic_search.search_type,
                                           search_kwargs={"k": max_k})
    
    chain = load_qa_chain(llm=llm, chain_type=chain_type, prompt=prompt)

    while True:
        question = input("\nENTER QUESTION >> ")
        output = get_and_parse_response(
            prompt=question, chain=chain, embed_retriever=embed_retriever, config=config.semantic_search
        )
        print_llm_response(output)
