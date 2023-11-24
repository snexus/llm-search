import argparse
import gc
import os
from io import StringIO
from pathlib import Path
from typing import List

import langchain
import streamlit as st
import torch
import yaml
from dotenv import load_dotenv
from loguru import logger
from streamlit import chat_message

from llmsearch.config import Config
from llmsearch.chroma import VectorStoreChroma
from llmsearch.process import get_and_parse_response
from llmsearch.utils import get_llm_bundle, set_cache_folder
from llmsearch.embeddings import update_embeddings, EmbeddingsHashNotExistError

st.set_page_config(page_title="LLMSearch", page_icon=":robot:", layout="wide")

load_dotenv()

langchain.debug = True


@st.cache_data
def parse_cli_arguments():
    parser = argparse.ArgumentParser(description="Web application for LLMSearch")
    parser.add_argument("--config_path", dest="cli_config_path", type=str, default="")
    try:
        args = parser.parse_args()
    except SystemExit as e:
        os._exit(e.code)
    return args


def hash_func(obj: Config) -> str:
    return str(obj.embeddings.embeddings_path)

def udpate_index(config_file: str ):
    """Updates index on-fly

    Args:
        config_file (str): _description_
    """

    with st.spinner("Updating index, please wait..."):
        logger.debug("Unloading model...")
        unload_model()
        config = load_config(config_file)
        set_cache_folder(str(config.cache_folder))

        vs = VectorStoreChroma(
            persist_folder=str(config.embeddings.embeddings_path),
            config=config
        )
        try:
            logger.debug("Updating embeddings")
            stats = update_embeddings(config, vs)
        except EmbeddingsHashNotExistError:
            st.error("Couldn't find hash files. Please re-create the index using current version of the app.")
        else:
            logger.info(stats)
        finally:
            logger.debug("Cleaning memory and re-Loading model...")

            vs.unload()

            vs = None

            gc.collect()
            with torch.no_grad():
                torch.cuda.empty_cache()

            reload_model(config_file=config_file)
    st.success("Done updating.")

@st.cache_data
def load_config(config_file):
    if isinstance(config_file, str):
        logger.info(f"Loading data from a file: {config_file}")
        with open(config_file, "r") as f:
            string_data = f.read()
    else:
        stringio = StringIO(config_file.getvalue().decode("utf-8"))
        string_data = stringio.read()

    config_dict = yaml.safe_load(string_data)
    return Config(**config_dict)


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


@st.cache_data
def generate_response(
    question: str, use_hyde: bool, use_multiquery, _config: Config, _bundle, label_filter: str = ""
):
    # _config and _bundle are under scored so paratemeters aren't hashed

    output = get_and_parse_response(
        query=question, config=_config, llm_bundle=_bundle, label=label_filter
    )
    return output


@st.cache_data
def get_config_paths(config_dir: str) -> List[str]:
    root = Path(config_dir)
    config_paths = sorted([str(p) for p in root.glob("*.yaml")])
    return config_paths


def reload_model(config_file: str):
    if st.session_state["disable_load"]:
        logger.info("In process of loading the model, please wait...")
        return
    st.session_state["disable_load"] = True
    # This is required for memory management, we need to try and unload the model before loading new one

    logger.info("Clearing state and re-loading model...")
    unload_model()

    logger.debug(f"Reload model got CONFIG FILE NAME: {config_file}")
    with st.spinner("Loading configuration"):
        config = load_config(config_file)
        st.session_state["llm_bundle"] = get_llm_bundle(config)
        st.session_state["llm_config"] = {"config": config, "file": config_file}

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

if Path(args.cli_config_path).is_dir():
    config_paths = get_config_paths(args.cli_config_path)
    config_file = st.sidebar.selectbox(
        label="Choose config", options=config_paths, index=0
    )
    logger.debug(f"CONFIG FILE: {config_file}")

    # Every form must have a submit button.
    load_button = st.sidebar.button("Load", on_click=reload_model, args=(config_file,), type="primary")

    # Since in the event loop on_click will be called first, we need to re-enable the flag in case of multiple clicks
    if load_button:
        st.session_state["disable_load"] = False



if st.session_state["llm_bundle"] is not None:
    config = st.session_state["llm_config"]["config"]

    config_file = st.session_state["llm_config"]["file"]
    config_file_name = config_file if isinstance(config_file, str) else config_file.name
    st.sidebar.subheader("Loaded Parameters:")
    with st.sidebar.expander(config_file_name):
        st.json(config.json())

    st.sidebar.write(f"**Model type:** {config.llm.type}")

    st.sidebar.write(
        f"**Document path**: {config.embeddings.document_settings[0].doc_path}"
    )
    st.sidebar.write(f"**Embedding path:** {config.embeddings.embeddings_path}")
    update_embeddings_button = st.sidebar.button("Update embeddings", on_click=udpate_index, args=(config_file,), type="secondary")
    st.sidebar.write(
        f"**Max char size (semantic search):** {config.semantic_search.max_char_size}"
    )
    label_filter = ""
    if config.embeddings.labels:
        label_filter = st.sidebar.selectbox(
            label="Filter by label", options=["-"] + config.embeddings.labels
        )
        if label_filter is None or label_filter == "-":
            label_filter = ""

    text = st.chat_input("Enter text", disabled=False)
    is_hyde = st.sidebar.checkbox(
        label="Use HyDE (cost: 2 api calls)",
        value=st.session_state["llm_bundle"].hyde_enabled,
    )
    is_multiquery = st.sidebar.checkbox(
        label="Use MultiQuery (cost: 2 api calls)",
        value=st.session_state["llm_bundle"].multiquery_enabled,
    )

    if text:
        # Dynamically switch hyde
        st.session_state["llm_bundle"].hyde_enabled = is_hyde
        st.session_state["llm_bundle"].multiquery_enabled = is_multiquery
        output = generate_response(
            question=text,
            use_hyde=st.session_state["llm_bundle"].hyde_enabled,
            use_multiquery=st.session_state["llm_bundle"].multiquery_enabled,
            _bundle=st.session_state["llm_bundle"],
            _config=config,
            label_filter=label_filter,
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
