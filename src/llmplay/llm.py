from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import (LangchainEmbedding, LLMPredictor, PromptHelper,
                         ServiceContext)

load_dotenv()

class LLMOpenAI:
    def __init__(self, embedding_model_name: str, cache_fodler: str, llm_model_name: str):
        self._llm_model_name = llm_model_name
        self._embedding_model_name = embedding_model_name
        self._service_context = None
        self.cache_folder = cache_fodler


    @property
    def service_context(self):
        if self._service_context is None:
            embed_model = LangchainEmbedding(
                HuggingFaceEmbeddings(model_name=self._embedding_model_name, cache_folder=self.cache_folder)
            )
            llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name=self._llm_model_name))

            max_input_size = 4096
            # set number of output tokens
            num_output = 2048
            # set maximum chunk overlap
            max_chunk_overlap = 20
            prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

            self._service_context = ServiceContext.from_defaults(
                llm_predictor=llm_predictor, embed_model=embed_model, chunk_size_limit=512, prompt_helper=prompt_helper
            )
        return self._service_context
