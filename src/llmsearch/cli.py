import os

import click
from loguru import logger

from llmsearch.chroma import VectorStoreChroma
from llmsearch.config import get_config
from llmsearch.interact import qa_with_llm
from llmsearch.models.utils import get_llm
from llmsearch.parsers.splitter import DocumentSplitter


@click.group(name="index")
def index_group():
    """Index generation commands."""


@click.group(name="interact")
def interact_group():
    """Commands to interact in Q&A sessiont with embedded content using LLMs"""


@click.group
def main_cli():
    pass


@click.command(name="create")
@click.option(
    "--config-file",
    "-c",
    "config_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Specifies YAML configuration file",
)
def generate_index(config_file: str):
    config = get_config(config_file)
    set_cache_folder(str(config.cache_folder))
    
    splitter = DocumentSplitter(config.embeddings.document_settings)
    all_docs = splitter.split()
    
    vs = VectorStoreChroma(persist_folder=str(config.embeddings.embeddings_path))
    vs.create_index_from_documents(all_docs=all_docs)
    


def set_cache_folder(cache_folder_root: str):
    sentence_transformers_home = cache_folder_root
    transformers_cache = os.path.join(cache_folder_root, "transformers")
    hf_home = os.path.join(cache_folder_root, "hf_home")

    logger.info(
        f"Setting SENTENCE_TRANSFORMERS_HOME folder: {sentence_transformers_home}"
    )
    logger.info(f"Setting TRANSFORMERS_CACHE folder: {transformers_cache}")
    logger.info(f"Setting HF_HOME: {hf_home}")
    logger.info(f"Setting MODELS_CACHE_FOLDER: {cache_folder_root}")

    os.environ["SENTENCE_TRANSFORMERS_HOME"] = sentence_transformers_home
    os.environ["TRANSFORMERS_CACHE"] = transformers_cache
    os.environ["HF_HOME"] = hf_home
    os.environ["MODELS_CACHE_FOLDER"] = cache_folder_root


@click.command("llm")
@click.option(
    "--config-file",
    "-c",
    "config_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Specifies YAML configuration file",
)
def launch_qa_with_llm(config_file: str):
    config = get_config(config_file)
    set_cache_folder(str(config.cache_folder))

    llm = get_llm(config.llm.params)  # type: ignore
    qa_with_llm(llm=llm.model, prompt=llm.prompt, config=config)


index_group.add_command(generate_index)
interact_group.add_command(launch_qa_with_llm)

# add command groups to CLI root
main_cli.add_command(index_group)
main_cli.add_command(interact_group)

if __name__ == "__main__":
    main_cli()
