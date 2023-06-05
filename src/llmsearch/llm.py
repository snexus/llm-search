from typing import Any, List, Mapping, Optional

import torch
from dotenv import load_dotenv
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.llms.base import LLM
from llama_index import LangchainEmbedding, LLMPredictor, PromptHelper, ServiceContext
from transformers import pipeline
import transformers
from transformers import StoppingCriteria, StoppingCriteriaList

load_dotenv()


class AbstractModel:
    def __init__(
        self,
        embedding_model_name: str,
        llm_model_name: str,
        max_input_size: int,
        num_output: int,
        max_chunk_overlap: int = 20,
        cache_folder=None,
    ):
        self._llm_model_name = llm_model_name
        self._embedding_model_name = embedding_model_name
        self._service_context = None
        self._cache_folder = cache_folder

        self._max_input_size = max_input_size
        self._num_output = num_output
        self._max_chunk_overlap = max_chunk_overlap

    @property
    def prompt_helper(self) -> PromptHelper:
        return PromptHelper(
            max_input_size=self._max_input_size, num_output=self._num_output, max_chunk_overlap=self._max_chunk_overlap
        )


class LLMOpenAIWrapper(AbstractModel):
    def __init__(self, temperature: int = 0, **kwargs):
        super().__init__(**kwargs)
        self._temperature = temperature

    @property
    def service_context(self):
        if self._service_context is None:
            embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name=self._embedding_model_name))
            llm_predictor = LLMPredictor(
                llm=ChatOpenAI(temperature=self._temperature, model_name=self._llm_model_name)
            )

            self._service_context = ServiceContext.from_defaults(
                llm_predictor=llm_predictor,
                embed_model=embed_model,
                chunk_size_limit=512,
                prompt_helper=self.prompt_helper,
            )
        return self._service_context


class CustomLLM(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.pipeline(prompt)
        return response[0]["generated_text"]  # Hardcoded for the time being

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "custom"

    @classmethod
    def from_custom_model(cls, model_name: str, pipeline_kwargs, model_kwargs):
        cls.model_name = model_name
        cls.pipeline = pipeline(
            model=model_name,
            **pipeline_kwargs,
            model_kwargs=model_kwargs,
        )
        return cls()


class HuggingFaceWrapper(AbstractModel):
    @property
    def service_context(self):
        if self._service_context is None:
            embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name=self._embedding_model_name))

            llm_predictor = LLMPredictor(
                llm=CustomLLM.from_custom_model(
                    model_name=self._llm_model_name, model_kwargs={"cache_dir": self._cache_folder}
                )
            )

            self._service_context = ServiceContext.from_defaults(
                llm_predictor=llm_predictor,
                embed_model=embed_model,
                chunk_size_limit=512,
                prompt_helper=self.prompt_helper,
            )
        return self._service_context


class LLMDatabricksDolly:
    pipeline_kwargs = dict(
        torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", return_full_text=True
    )

    def __init__(self, cache_folder, model_name="databricks/dolly-v2-3b") -> None:
        self.cache_folder = cache_folder
        self.model_name = model_name

    @property
    def model(self):
        llm = CustomLLM.from_custom_model(
            model_name=self.model_name,
            pipeline_kwargs=self.pipeline_kwargs,
            model_kwargs={"cache_dir": self.cache_folder, "max_length": 2048},
        )

        return llm


class LLMDatabricksDollyV2:
    pipeline_kwargs = dict(
        torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", return_full_text=True
    )

    def __init__(self, cache_folder, model_name="databricks/dolly-v2-3b") -> None:
        self.cache_folder = cache_folder
        self.model_name = model_name

    @property
    def model(self):
        generate_text = pipeline(
            model=self.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            return_full_text=True,
            model_kwargs={"cache_dir": self.cache_folder},
        )

        hf_pipeline = HuggingFacePipeline(pipeline=generate_text)
        return hf_pipeline


class LLMMosaicMPT:
    def __init__(self, cache_folder, model_name="mosaicml/mpt-7b-chat", device = "auto") -> None:
        self.cache_folder = cache_folder
        self.model_name = model_name
        self.device = device

    @property
    def model(self):
        # https://github.com/pinecone-io/examples/blob/master/generation/llm-field-guide/mpt-7b/mpt-7b-huggingface-langchain.ipynb
        
        if self.device == "auto":
            device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device

        model = transformers.AutoModelForCausalLM.from_pretrained(
            "mosaicml/mpt-7b-instruct",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            max_seq_len=2048,
            cache_dir=self.cache_folder,
        )
        
        model.eval()
        model.to(device)
        
        tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

        stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>"])

        # define custom stopping criteria object
        class StopOnTokens(StoppingCriteria):
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                for stop_id in stop_token_ids:
                    if input_ids[0][-1] == stop_id:
                        return True
                return False

        stopping_criteria = StoppingCriteriaList([StopOnTokens()])

        generate_text = transformers.pipeline(
            model=model,
            tokenizer=tokenizer,
            return_full_text=True,  # langchain expects the full text
            task="text-generation",
            #device=device,
            device_map="auto",
            # we pass model parameters here too
            stopping_criteria=stopping_criteria,  # without this model will ramble
            temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            #top_p=0.15,  # select from top tokens whose probability add up to 15%
            #top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
            max_new_tokens=1024,  # mex number of tokens to generate in the output
           # repetition_penalty=1.1,  # without this output begins repeating
            model_kwargs={"cache_dir": self.cache_folder},
        )

        hf_pipeline = HuggingFacePipeline(pipeline=generate_text)
        return hf_pipeline
