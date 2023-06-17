import os

import click
from loguru import logger

from llmsearch.interact import qa_with_llm
from llmsearch.llm import get_llm_model
from llmsearch.llm import ModelConfig
from llmsearch.chroma import VectorStoreChroma


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
    "--document-folder",
    "-d",
    "document_folder",
    required=True,
    type=click.Path(exists=True, dir_okay=True, writable=True),
    help="Specifies a document folder to scan.",
)
@click.option(
    "--output-embedding-folder",
    "-o",
    "output_embedding_folder",
    required=True,
    type=click.Path(exists=True, dir_okay=True, writable=True),
    help="Specifies a folder to store the embeddings.",
)
@click.option(
    "--cache-folder",
    "-c",
    "cache_folder_root",
    required=True,
    type=click.Path(exists=True, dir_okay=True, writable=True),
    help="Specifies a cache folder",
)
@click.option(
    "--embedding-model-name", "-m", "embed_model_name", default="all-MiniLM-L6-v2", help="Specifies HF embedding model"
)
@click.option(
    "--scan-extension",
    "-e",
    "scan_extension",
    default="md",
    help="Specifies an file extension to scan, for example `md`",
)
def generate_index(
    document_folder, output_embedding_folder, cache_folder_root: str, embed_model_name: str = "all-MiniLM-L6-v2", scan_extension: str = "md"
):
    set_cache_folder(cache_folder_root)
    vs = VectorStoreChroma(persist_folder=output_embedding_folder, hf_embed_model_name=embed_model_name)
    vs.create_index_from_folder(folder_path=document_folder, extension=scan_extension)


def set_cache_folder(cache_folder_root: str):
    sentence_transformers_home = cache_folder_root
    transformers_cache = os.path.join(cache_folder_root, "transformers")
    hf_home = os.path.join(cache_folder_root, "hf_home")


    logger.info(f"Setting SENTENCE_TRANSFORMERS_HOME folder: {sentence_transformers_home}")
    logger.info(f"Setting TRANSFORMERS_CACHE folder: {transformers_cache}")
    logger.info(f"Setting HF_HOME: {hf_home}")
    logger.info(f"Setting MODELS_CACHE_FOLDER: {cache_folder_root}")

    os.environ["SENTENCE_TRANSFORMERS_HOME"] = sentence_transformers_home
    os.environ["TRANSFORMERS_CACHE"] = transformers_cache
    os.environ["HF_HOME"] = hf_home
    os.environ["MODELS_CACHE_FOLDER"] = cache_folder_root



@click.command("llm")
@click.option(
    "--embeddings-folder",
    "-f",
    "embedding_persist_folder",
    required=True,
    type=click.Path(exists=True, dir_okay=True),
    help="Folders where embeddings are stored",
)
@click.option(
    "--llm-model-name",
    "-m",
    "model_name",
    required=True,
    type=click.Choice([model.value for model in ModelConfig], case_sensitive=True),
    help="Choice of available LLM models",
)
@click.option(
    "--cache-folder",
    "-c",
    "cache_folder_root",
    required=True,
    type=click.Path(exists=True, dir_okay=True, writable=True),
    help="Specifies a cache folder",
)
@click.option(
    "--quant-8bit",
    "-q8",
    "is_8bit",
    default=False,
    type=bool,
    help="Turns on 8bit quantization. Significantly reduces memory usage.",
)
@click.option(
    "--embedding-model-name",
    "-e",
    "embedding_model_name",
    default="all-MiniLM-L6-v2",
    help="Specifies HF embedding model",
)
@click.option(
    "--chain-type",
    "-t",
    "chain_type",
    default="stuff",
    help="Specifies how nodes are chained together, when passed to LLM",
)

@click.option(
    "--max-context-size",
    "-cs",
    "max_context_size",
    type = click.IntRange(min=512, max=4096),
    default= 2048,
    help="Specifies max context size",
)
@click.option(
    "--model-path",
    "-mf",
    "model_path",
    required=False,
    type=click.Path(exists=True, dir_okay=True),
    help="Specifies a folder location of quantized GPTQ model or a path to a model for GGML model",
)
def launch_qa_with_llm(
    embedding_persist_folder: str,
    model_name: str,
    cache_folder_root: str,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    chain_type="stuff",
    is_8bit = False,
    max_context_size = 2048, 
    model_path = None
):

    set_cache_folder(cache_folder_root)
    model_cache_folder = os.environ.get("MODELS_CACHE_FOLDER")
    
    logger.info(f"Invoking Q&A tool using {model_name} LLM")
    llm_settings = get_llm_model(model_name, cache_folder_root = model_cache_folder, is_8bit = is_8bit, model_path=model_path)
    qa_with_llm(
        embedding_persist_folder=embedding_persist_folder,
        llm=llm_settings.llm,
        prompt = llm_settings.prompt,
        embedding_model_name=embedding_model_name,
        chain_type=chain_type,
         max_context_size = max_context_size
    )


index_group.add_command(generate_index)
interact_group.add_command(launch_qa_with_llm)

# add command groups to CLI root
main_cli.add_command(index_group)
main_cli.add_command(interact_group)

if __name__ == "__main__":
    main_cli()
