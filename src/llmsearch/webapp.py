"""Web application for LLMSearch"""

import argparse
import gc
import io
import os
from io import StringIO
from pathlib import Path
from typing import Dict, Union

import langchain
import streamlit as st
import torch
import yaml
from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from loguru import logger
from streamlit import chat_message

from llmsearch.chroma import VectorStoreChroma
from llmsearch.config import Config
from llmsearch.embeddings import (
    EmbeddingsHashNotExistError,
    create_embeddings,
    load_document_labels,
    update_embeddings,
)
from llmsearch.process import get_and_parse_response
from llmsearch.utils import get_llm_bundle, set_cache_folder

st.set_page_config(page_title="LLMSearch", page_icon=":robot:", layout="wide")

load_dotenv()

langchain.debug = True  # type: ignore

# See https://github.com/VikParuchuri/marker/issues/442
torch.classes.__path__ = []  # add this line to manually set it to empty.


@st.cache_data
def parse_cli_arguments():
    parser = argparse.ArgumentParser(description="Web application for LLMSearch")
    parser.add_argument(
        "--doc_config_path", dest="cli_doc_config_path", type=str, default=""
    )
    parser.add_argument(
        "--model_config_path", dest="cli_model_config_path", type=str, default=""
    )
    try:
        args = parser.parse_args()
    except SystemExit as e:
        os._exit(e.code)
    return args


def hash_func(obj: Config) -> str:
    return str(obj.embeddings.embeddings_path)


def generate_index(config: Config):
    with st.spinner("Creading index, please wait..."):
        logger.debug("Unloading existing models...")
        unload_model()
        set_cache_folder(str(config.cache_folder))

        vs = VectorStoreChroma(
            persist_folder=str(config.embeddings.embeddings_path), config=config
        )
        create_embeddings(config, vs)
        logger.debug("Cleaning memory and re-Loading model...")
        vs.unload()
        vs = None
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()

    st.success("Done generating index.")


def udpate_index(doc_config_path: str, model_config_file):
    """Updates index on-fly

    Args:
        config_file (str): _description_
    """

    with st.spinner("Updating index, please wait..."):
        logger.debug("Unloading model...")
        unload_model()
        conf = load_config(doc_config_path, model_config_file)
        set_cache_folder(str(conf.cache_folder))

        vs = VectorStoreChroma(
            persist_folder=str(conf.embeddings.embeddings_path), config=conf
        )
        try:
            logger.debug("Updating embeddings")
            stats = update_embeddings(conf, vs)
        except EmbeddingsHashNotExistError:
            st.error(
                "Couldn't find hash files. Please re-create the index using current version of the app."
            )
        else:
            logger.info(stats)
        finally:
            logger.debug("Cleaning memory and re-Loading model...")

            vs.unload()

            vs = None

            gc.collect()
            with torch.no_grad():
                torch.cuda.empty_cache()

            reload_model(
                doc_config_path=doc_config_path, model_config_file=model_config_file
            )
    st.success("Done updating.")


@st.cache_data
def load_config(doc_config, model_config) -> Config:
    """Loads doc and model configurations, combines, and returns an instance of Config"""

    doc_config_dict = load_yaml_file(doc_config)
    model_config_dict = load_yaml_file(model_config)

    config_dict = {**doc_config_dict, **model_config_dict}
    return Config(**config_dict)


@st.cache_data
def load_labels(embedding_path: str) -> Dict[str, str]:
    labels_fn = Path(os.path.join(embedding_path, "labels.txt"))
    all_labels = {Path(label).name: label for label in load_document_labels(labels_fn)}
    return all_labels


@st.cache_data
def load_yaml_file(conf: Union[str, io.BytesIO]) -> dict:
    """Loads YAML file and returns a dictionary"""
    if isinstance(conf, str):
        logger.info(f"Loading doc config from a file: {conf}")
        with open(conf, "r", encoding="utf-8") as f:
            string_data = f.read()
    else:
        stringio = StringIO(conf.getvalue().decode("utf-8"))
        string_data = stringio.read()

    config_dict = yaml.safe_load(string_data)
    return config_dict


def unload_model():
    """Unloads llm_bundle from the state to free up the GPU memory"""

    if st.session_state["llm_bundle"] is not None:
        st.session_state["llm_bundle"].store = None
        st.session_state["llm_bundle"].chain = None
        st.session_state["llm_bundle"].reranker = None
        st.session_state["llm_bundle"].hyde_chain = None
        st.session_state["llm_bundle"].multiquery_chain = None

    st.cache_data.clear()
    st.cache_resource.clear()
    gc.collect()

    st.session_state["llm_bundle"] = None
    gc.collect()

    with torch.no_grad():
        torch.cuda.empty_cache()


def clear_chat_history():
    if st.session_state["llm_bundle"] is not None:
        st.session_state["llm_bundle"].conversation_history_settings.history = []

        # Clear LLM Cache
        st.session_state["llm_bundle"].llm_cache.clear()

        # Clear Streamlit Cache
        st.cache_data.clear()


@st.cache_data
def generate_response(
    question: str,
    use_hyde: bool,
    use_multiquery,
    _config: Config,
    _bundle,
    label_filter: str = "",
    source_chunk_type_filter: str = "",
):
    # _config and _bundle are under scored so paratemeters aren't hashed

    output = get_and_parse_response(
        query=question,
        config=_config,
        llm_bundle=_bundle,
        label=label_filter,
        source_chunk_type=source_chunk_type_filter,
    )
    return output


@st.cache_data
def get_config_paths(config_dir: str) -> Dict[str, str]:
    root = Path(config_dir)
    # config_paths = sorted([p for p in root.glob("*.yaml")])
    config_path_names = {p.name: str(p) for p in root.glob("*.yaml")}
    return config_path_names


def reload_model(doc_config_path: str, model_config_file: str):
    if st.session_state["disable_load"]:
        logger.info("In process of loading the model, please wait...")
        return
    st.session_state["disable_load"] = True
    # This is required for memory management, we need to try and unload the model before loading new one

    logger.info("Clearing state and re-loading model...")
    unload_model()

    logger.debug(f"Reload model got DOC CONFIG FILE NAME: {doc_config_path}")
    logger.debug(f"Reload model got MODEL CONFIG FILE NAME: {model_config_file}")
    with st.spinner("Loading configuration"):
        conf = load_config(doc_config_path, model_config_file)
        if conf.check_embeddings_exist():
            st.session_state["llm_bundle"] = get_llm_bundle(conf)
            st.session_state["llm_config"] = {
                "config": conf,
                "doc_config_path": doc_config_path,
                "model_config_file": model_config_file,
            }
        else:
            st.error(
                "Couldn't find embeddings in {}. Please generate first.".format(
                    conf.embeddings.embeddings_path
                )
            )
            _ = st.button(
                "Generate", on_click=generate_index, args=(conf,), type="secondary"
            )

    logger.debug("Setting LLM Cache")
    set_llm_cache(st.session_state["llm_bundle"].llm_cache)
    st.session_state["disable_load"] = False


st.title(":sparkles: LLMSearch")
args = parse_cli_arguments()
st.sidebar.subheader(":hammer_and_wrench: Configuration")

# Initialsize state for historical resutls
if "llm_bundle" not in st.session_state:
    st.session_state["llm_bundle"] = None

if "llm_config" not in st.session_state:
    st.session_state["llm_config"] = None

# Initialsize state for historical resutls
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "disable_load" not in st.session_state:
    st.session_state["disable_load"] = False

if Path(args.cli_doc_config_path).is_dir():
    config_paths = get_config_paths(args.cli_doc_config_path)
    doc_config_name = st.sidebar.selectbox(
        label="Choose config", options=sorted(list(config_paths.keys())), index=0
    )
    doc_config_path = config_paths[doc_config_name]  # type: ignore
    model_config_file = args.cli_model_config_path
    logger.debug(f"CONFIG FILE: {doc_config_path}")

    # Every form must have a submit button.
    load_button = st.sidebar.button(
        "Load",
        on_click=reload_model,
        args=(doc_config_path, model_config_file),
        type="primary",
    )

    # Since in the event loop on_click will be called first, we need to re-enable the flag in case of multiple clicks
    if load_button:
        st.session_state["disable_load"] = False


if st.session_state["llm_bundle"] is not None:
    config = st.session_state["llm_config"]["config"]

    doc_config_path = st.session_state["llm_config"]["doc_config_path"]
    model_config_file = st.session_state["llm_config"]["model_config_file"]
    config_file_name = (
        doc_config_path if isinstance(doc_config_path, str) else doc_config_path.name
    )
    st.sidebar.subheader("Loaded Parameters:")
    with st.sidebar.expander(config_file_name):
        st.json(config.model_dump_json())

    st.sidebar.write(f"**Model type:** {config.llm.type}")

    st.sidebar.write(
        f"**Document path**: {config.embeddings.document_settings[0].doc_path}"
    )
    st.sidebar.write(f"**Embedding path:** {config.embeddings.embeddings_path}")
    update_embeddings_button = st.sidebar.button(
        "Update embeddings",
        on_click=udpate_index,
        args=(doc_config_path, model_config_file),
        type="secondary",
    )
    st.sidebar.write(
        f"**Max char size (semantic search):** {config.semantic_search.max_char_size}"
    )
    label_filter = ""
    document_labels = load_labels(config.embeddings.embeddings_path)

    if document_labels:
        label_filter = st.sidebar.selectbox(
            label="Restrict search to:",
            options=["-"] + sorted(list(document_labels.keys())),
        )
        if label_filter is None or label_filter == "-":
            label_filter = ""

    # tables_only_filter = st.sidebar.checkbox(label="Prioritize tables")
    # if tables_only_filter:
    # source_chunk_type_filter="table"
    # else:
    # source_chunk_type_filter=""

    source_chunk_type_filter = ""

    text = st.chat_input("Enter text", disabled=False)
    is_hyde = st.sidebar.checkbox(
        label="Use HyDE (LLM calls: 2)",
        value=st.session_state["llm_bundle"].hyde_enabled,
    )
    is_multiquery = st.sidebar.checkbox(
        label="Use MultiQuery (LLM calls: 2)",
        value=st.session_state["llm_bundle"].multiquery_enabled,
    )
    conv_history_enabled = st.sidebar.checkbox(
        label="Enable follow-up questions",
        value=st.session_state["llm_bundle"].conversation_history_settings.enabled,
    )

    if conv_history_enabled:
        conv_history_max_length = st.sidebar.number_input(
            "Maximum history length (QA pairs)",
            min_value=1,
            value=st.session_state[
                "llm_bundle"
            ].conversation_history_settings.max_history_length,
        )
        conv_history_rewrite_query = st.sidebar.checkbox(
            label="Contextualize user question (LLM calls: 2)", value=True
        )

        clear_conv_history = st.sidebar.button(
            "Clear history", on_click=clear_chat_history, type="secondary"
        )

    if text:
        # Dynamically switch hyde
        st.session_state["llm_bundle"].hyde_enabled = is_hyde
        st.session_state["llm_bundle"].multiquery_enabled = is_multiquery
        st.session_state["llm_bundle"].conversation_history_settings.enabled = (
            conv_history_enabled
        )

        # Set conversation history settings
        if conv_history_enabled:

            st.session_state[
                "llm_bundle"
            ].conversation_history_settings.max_history_length = conv_history_max_length
            st.session_state[
                "llm_bundle"
            ].conversation_history_settings.rewrite_query = conv_history_rewrite_query

        output = generate_response(
            question=text,
            use_hyde=st.session_state["llm_bundle"].hyde_enabled,
            use_multiquery=st.session_state["llm_bundle"].multiquery_enabled,
            _bundle=st.session_state["llm_bundle"],
            _config=config,
            label_filter=document_labels.get(label_filter, ""),
            source_chunk_type_filter=source_chunk_type_filter,
        )

        # Add assistant response to chat history
        st.session_state["messages"].append(
            {
                "question": text,
                "response": output.response,
                "links": [
                    f'<a href="{s.chunk_link}">{s.chunk_link}</a>'
                    for s in output.semantic_search[::-1]
                ],
                "quality": f"{output.average_score:.2f}",
            }
        )
        for h_response in st.session_state["messages"]:
            with st.expander(
                label=f":question: **{h_response['question']}**", expanded=False
            ):
                st.markdown(f"##### {h_response['question']}")
                st.write(h_response["response"])
                st.markdown(
                    f"\n---\n##### Serrch Quality Score: {h_response['quality']}"
                )
                st.markdown("##### Links")
                for link in h_response["links"]:
                    st.write("\t* " + link, unsafe_allow_html=True)

        for source in output.semantic_search[::-1]:
            source_path = source.metadata.pop("source")
            score = source.metadata.get("score", None)
            with st.expander(label=f":file_folder: {source_path}", expanded=True):
                st.write(
                    f'<a href="{source.chunk_link}">{source.chunk_link}</a>',
                    unsafe_allow_html=True,
                )
                if score is not None:
                    st.write(f"\nScore: {score}")

                st.text(f"\n\n{source.chunk_text}")
        if st.session_state["llm_bundle"].hyde_enabled:
            with st.expander(label=":octagonal_sign: **HyDE Reponse**", expanded=False):
                st.write(output.hyde_response)

        with chat_message("assistant"):
            st.write(f"**Search results quality score: {output.average_score:.2f}**\n")
            st.write(output.response)  # Add user message to chat history


else:
    st.info("Choose a configuration template to start...")
