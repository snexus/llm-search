import os

import click
import streamlit.web.cli
from loguru import logger

from llmsearch.chroma import VectorStoreChroma
from llmsearch.config import get_config, get_doc_with_model_config
from llmsearch.embeddings import create_embeddings, update_embeddings
from llmsearch.interact import qa_with_llm
from llmsearch.utils import get_llm_bundle, set_cache_folder


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

    vs = VectorStoreChroma(
        persist_folder=str(config.embeddings.embeddings_path), config=config
    )
    create_embeddings(config, vs)


@click.command(name="update")
@click.option(
    "--config-file",
    "-c",
    "config_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Specifies YAML configuration file",
)
def udpate_index(config_file: str):
    config = get_config(config_file)
    set_cache_folder(str(config.cache_folder))

    vs = VectorStoreChroma(
        persist_folder=str(config.embeddings.embeddings_path), config=config
    )
    stats = update_embeddings(config, vs)
    logger.info(stats)


@click.command("llm")
@click.option(
    "--doc-config-file",
    "-c",
    "doc_config_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    help="Specifies documents YAML configuration file",
)
@click.option(
    "--model-config-file",
    "-m",
    "model_config_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    help="Specifies model YAML configuration file",
)
def launch_qa_with_llm(doc_config_file: str, model_config_file: str):
    config = get_doc_with_model_config(doc_config_file, model_config_file)
    llm_bundle = get_llm_bundle(config)
    qa_with_llm(llm_bundle, config)


@click.command("webapp")
@click.option(
    "--doc-config-path",
    "-c",
    "doc_config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Specifies documents YAML configuration file",
)
@click.option(
    "--model-config-file",
    "-m",
    "model_config_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
    help="Specifies model YAML configuration file",
)
def launch_streamlit(doc_config_path: str, model_config_file: str):
    # Based on
    # https://discuss.streamlit.io/t/running-streamlit-inside-my-own-executable-with-the-click-module/1198/4
    # streamlit run ./src/llmsearch/webapp.py -- --config_path ./sample_templates/obsidian_conf.yaml

    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "webapp.py")
    args = [
        "--doc_config_path",
        doc_config_path,
        "--model_config_path",
        model_config_file,
    ]
    streamlit.web.cli._main_run(filename, args)


index_group.add_command(generate_index)
index_group.add_command(udpate_index)
interact_group.add_command(launch_qa_with_llm)

index_group.add_command(generate_index)
interact_group.add_command(launch_qa_with_llm)
interact_group.add_command(launch_streamlit)

# add command groups to CLI root
main_cli.add_command(index_group)
main_cli.add_command(interact_group)

if __name__ == "__main__":
    main_cli()
