from io import StringIO
from streamlit import chat_message, chat_input
from llmsearch.cli import set_cache_folder
from llmsearch.config import get_config
from llmsearch.interact import qa_with_llm
from llmsearch.models.utils import get_llm

import langchain
import streamlit as st
import yaml
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from termcolor import cprint

from llmsearch.chroma import VectorStoreChroma
from llmsearch.config import Config, ResponseModel 
from llmsearch.process import get_and_parse_response

st.set_page_config(page_title="LLMSearch", page_icon=":robot:", layout="wide")
load_dotenv()

MAX_K = 7
CHAIN_TYPE="stuff"


def hash_func(obj: Config) -> str:
    return str(obj.embeddings.embeddings_path)

@st.cache_data
def load_config(config_file):
    stringio = StringIO(config_file.getvalue().decode("utf-8"))
    string_data = stringio.read()

    config_dict = yaml.safe_load(string_data)
    return Config(**config_dict)

@st.cache_resource(hash_funcs={Config:hash_func})
def get_chain(config: Config):
    set_cache_folder(str(config.cache_folder))
    llm = get_llm(config.llm.params)
    chain  = load_qa_chain(llm = llm.model, chain_type=CHAIN_TYPE, prompt = llm.prompt)
    return chain

@st.cache_resource(hash_funcs={Config:hash_func})
def get_retriever(config):
    store = VectorStoreChroma(persist_folder=str(config.embeddings.embeddings_path))
    embed_retriever = store.load_retriever(
        search_type=config.semantic_search.search_type, search_kwargs={"k": MAX_K}
    )
    return embed_retriever
    

@st.cache_data
def generate_response(question: str, _config, _chain, _retriever):
    output = get_and_parse_response(
        query=question,
        chain=_chain,
        embed_retriever=_retriever,
        config=_config.semantic_search,
    )
    return output
    

st.title(":sparkles: LLMSearch")

st.sidebar.subheader(":hammer_and_wrench: Configuration")
config_file = st.sidebar.file_uploader("Select tempate to load", type=['yml','yaml'])


if config_file is not None:
    config = load_config(config_file) 
    

    with st.sidebar.expander(config_file.name):
        st.json(config.json())

    st.sidebar.write(f"**Model type:** {config.llm.type}")
    
    st.sidebar.write(f"**Docuemnt path**: {config.embeddings.doc_path}")
    st.sidebar.write(f"**Embedding path:** {config.embeddings.embeddings_path}")
    st.sidebar.write(f"**Max char size (semantic search):** {config.semantic_search.max_char_size}")


    chain = get_chain(config)
    retriever = get_retriever(config)

    text = st.chat_input('Enter text')
    # with st.form("my_form"):
        # submitted = st.form_submit_button("Get answer")
    if text:
        output = generate_response(text, config, chain, retriever)
        
        with chat_message("assistant"):
            st.write(output.response)
        
        
        for source in output.semantic_search:
            source_path = source.metadata.pop("source")
            with st.expander(label=f":file_folder: {source_path}", expanded=True):
                st.write(f'<a href="{source.chunk_link}">{source.chunk_link}</a>', unsafe_allow_html=True)
                st.text(f"\n\n{source.chunk_text}")
                    
else:
    st.info("Choose a configuration template to start...")