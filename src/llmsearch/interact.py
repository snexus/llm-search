import langchain
from dotenv import load_dotenv
from termcolor import cprint

from llmsearch.config import Config, ResponseModel
from llmsearch.process import get_and_parse_response
from llmsearch.utils import LLMBundle

load_dotenv()
langchain.debug = True


def print_llm_response(output: ResponseModel):
    print("\n============= SOURCES ==================")
    for source in output.semantic_search:
        source.metadata.pop("source")
        cprint(source.chunk_link, "blue")
        cprint(source.metadata, "cyan")
        print("******************* BEING EXTRACT *****************")
        print(f"{source.chunk_text}\n")
    print("\n============= RESPONSE =================")
    cprint(output.response, "red")
    print("------------------------------------------")


def qa_with_llm(llm_bundle: LLMBundle, config: Config):
    while True:
        question = input("\nENTER QUESTION >> ")
        output = get_and_parse_response(
            query=question,
            chain=llm_bundle.chain,
            retrievers=llm_bundle.retrievers,
            config=config.semantic_search,
            reranker=llm_bundle.reranker,
        )
        print_llm_response(output)
