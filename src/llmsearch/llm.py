import enum
from abc import ABC, abstractmethod
from collections import namedtuple
from pathlib import Path
from typing import Any, List, Mapping, Optional, Union


import torch
import transformers
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from dotenv import load_dotenv
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.llms.base import LLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextGenerationPipeline,
    pipeline,
)

from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


from llmsearch.prompts import (
    DOLLY_PROMPT_TEMPLATE,
    OPENAI_PROMPT_TEMPLATE,
    REDPAJAMA_TEMPLATE,
    TULU8_TEMPLATE,
    LLAMA_TEMPLATE,
    WIZARDLM10_TEMPLATE,
)

load_dotenv()


class ModelConfig(enum.Enum):
    OPENAI_GPT35 = "openai-gpt35"
    DOLLY3B = "databricks-dolly3b"
    DOLLY7B = "databricks-dolly7b"
    MPT7B = "mosaic-mpt7b-instruct"
    FALCON7B = "falcon-7b-instruct"
    GPTQTULU7B = "gptq-tulu-7b"
    REDPAJAMAINCITE = "redpajama-incite-7b"
    WIZARDLM = "wizardlm-1.0-ggml"
    NOUSHERMES = "nous-hermes-ggml"


# USed to group llm settings for the caller
LLMSettings = namedtuple("LLMSettings", "llm prompt")


def get_llm_model(
    model_name: str,
    cache_folder_root: Union[str, Path],
    is_8bit: bool,
    model_path: Optional[Union[str, Path]] = None,
):
    model = ModelConfig(model_name)

    prompt = None

    if model == ModelConfig.OPENAI_GPT35:
        model_instance = LLMOpenAI()

    elif model == ModelConfig.DOLLY3B:
        model_instance = LLMDatabricksDollyV2(
            cache_folder=cache_folder_root, model_name="databricks/dolly-v2-3b", load_8bit=is_8bit
        )

    elif model == ModelConfig.DOLLY7B:
        model_instance = LLMDatabricksDollyV2(
            cache_folder=cache_folder_root, model_name="databricks/dolly-v2-7b", load_8bit=is_8bit
        )

    elif model == ModelConfig.MPT7B:
        model_instance = LLMMosaicMPT(
            cache_folder=cache_folder_root, model_name="mosaicml/mpt-7b-instruct", load_8bit=is_8bit
        )

    elif model == ModelConfig.FALCON7B:
        model_instance = LLMFalcon(
            cache_folder=cache_folder_root, model_name="tiiuae/falcon-7b-instruct", load_8bit=is_8bit
        )
    elif model == ModelConfig.REDPAJAMAINCITE:
        model_instance = RedPajamaIncite(
            cache_folder=cache_folder_root,
            model_name="togethercomputer/RedPajama-INCITE-7B-Instruct",
            load_8bit=is_8bit,
        )
    elif model == ModelConfig.GPTQTULU7B:
        if model_path is None:
            raise SystemError("Specify `--model-path` for GPTQ models.")
        model_instance = BlokeTulu(
            cache_folder=cache_folder_root,
            model_name="TheBloke/tulu-7B-GPTQ",
            load_8bit=is_8bit,
            quantized_model_folder=model_path,
        )
    elif model == ModelConfig.WIZARDLM:
        if model_path is None:
            raise SystemError("Specify `--model-path` for WIZARDLM models.")
        model_instance = LlamaCPPModel(
            cache_folder=cache_folder_root,
            model_name="WizardLM",
            model_path=model_path,
            prompt_template=LLAMA_TEMPLATE,
            llama_kwargs=dict(n_ctx=1024, max_tokens=512, temperature=0.0, n_gpu_layers=35, n_batch=512),
        )
    elif model == ModelConfig.NOUSHERMES:
        if model_path is None:
            raise SystemError("Specify `--model-path` for NOUS HERMES model.")
        model_instance = LlamaCPPModel(
            cache_folder=cache_folder_root,
            model_name="NousHermes",
            model_path=model_path,
            prompt_template=LLAMA_TEMPLATE,
            llama_kwargs=dict(n_ctx=1024, max_tokens=512, temperature=0.0, n_gpu_layers=35, n_batch=512),
        )

    else:
        raise TypeError(f"Invalid model type. Got {model_name}")
    return LLMSettings(llm=model_instance.model, prompt=model_instance.prompt)


class AbstractLLMModel(ABC):
    def __init__(
        self,
        model_name: str,
        cache_folder: Union[Path, str],
        prompt_template: Optional[str] = None,
        load_8bit: Optional[bool] = False,
    ) -> None:
        self.model_name = model_name
        self.cache_folder = cache_folder
        self.prompt_template = prompt_template
        self.load_8bit = load_8bit

    @property
    @abstractmethod
    def model(self):
        raise NotImplemented

    @property
    def prompt(self) -> Optional[PromptTemplate]:
        if self.prompt_template:
            return PromptTemplate(input_variables=["context", "question"], template=self.prompt_template)
        return None


class LLMOpenAI(AbstractLLMModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            cache_folder="",
            model_name="chatgpt-3.5",
            prompt_template=OPENAI_PROMPT_TEMPLATE,
        )

    @property
    def model(self):
        return ChatOpenAI(temperature=0.0)


class LLMDatabricksDollyV2(AbstractLLMModel):
    def __init__(self, cache_folder, model_name="databricks/dolly-v2-3b", load_8bit=False) -> None:
        super().__init__(
            cache_folder=cache_folder,
            model_name=model_name,
            prompt_template=DOLLY_PROMPT_TEMPLATE,
            load_8bit=load_8bit,
        )

    @property
    def model(self):
        generate_text = pipeline(
            model=self.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            return_full_text=True,
            model_kwargs={"cache_dir": self.cache_folder, "load_in_8bit": self.load_8bit},
        )

        hf_pipeline = HuggingFacePipeline(pipeline=generate_text)
        return hf_pipeline


class LLMMosaicMPT(AbstractLLMModel):
    def __init__(self, cache_folder, model_name="mosaicml/mpt-7b-instruct", load_8bit=False, device="auto") -> None:
        super().__init__(
            cache_folder=cache_folder,
            model_name=model_name,
            prompt_template=DOLLY_PROMPT_TEMPLATE,
            load_8bit=load_8bit,
        )
        self.device = device

    @property
    def model(self):
        # https://github.com/pinecone-io/examples/blob/master/generation/llm-field-guide/mpt-7b/mpt-7b-huggingface-langchain.ipynb

        if self.device == "auto":
            device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device

        config = transformers.AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        config.attn_config["attn_impl"] = "triton"
        config.init_device = device

        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name,
            config=config,
            torch_dtype=torch.bfloat16,
            cache_dir=self.cache_folder,
            trust_remote_code=True,
            device_map="auto",
            load_in_8bit=self.load_8bit,
        )

        tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", device=device)

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
            config=config,
            tokenizer=tokenizer,
            return_full_text=True,  # langchain expects the full text
            task="text-generation",
            # device=device,
            device_map="auto",
            # we pass model parameters here too
            stopping_criteria=stopping_criteria,  # without this model will ramble
            temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            # top_p=0.15,  # select from top tokens whose probability add up to 15%
            # top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
            max_new_tokens=256,  # mex number of tokens to generate in the output
            # repetition_penalty=1.1,  # without this output begins repeating
            model_kwargs={"cache_dir": self.cache_folder},
        )

        hf_pipeline = HuggingFacePipeline(pipeline=generate_text)
        return hf_pipeline


class LLMFalcon(AbstractLLMModel):
    def __init__(self, cache_folder, model_name="tiiuae/falcon-7b-instruct", load_8bit=False, device="auto") -> None:
        super().__init__(
            cache_folder=cache_folder,
            model_name=model_name,
            prompt_template=DOLLY_PROMPT_TEMPLATE,
            load_8bit=load_8bit,
        )
        self.device = device

    @property
    def model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model_kwargs = {"temperature": 0.01, "cache_dir": self.cache_folder, "load_in_8bit": self.load_8bit}

        pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_name,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            max_new_tokens=512,
            # max_length=512,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            model_kwargs=model_kwargs,
        )

        llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs=model_kwargs)
        return llm


class BlokeTulu(AbstractLLMModel):
    def __init__(
        self,
        cache_folder: str,
        quantized_model_folder: str,
        model_name="TheBloke/tulu-7B-GPTQ",
        load_8bit=False,
        device="auto",
    ) -> None:
        super().__init__(
            cache_folder=cache_folder,
            model_name=model_name,
            prompt_template=DOLLY_PROMPT_TEMPLATE,
            load_8bit=load_8bit,
        )
        self.device = device
        self.model_base_name = "gptq_model-4bit-128g"
        self.quantized_model_folder = quantized_model_folder

    @property
    def model(self):
        if self.device == "auto":
            device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device

        tokenizer = AutoTokenizer.from_pretrained(self.quantized_model_folder, use_fast=True, device=device)

        model = AutoGPTQForCausalLM.from_quantized(
            self.quantized_model_folder,
            model_basename=self.model_base_name,
            use_safetensors=True,
            trust_remote_code=False,
            device=device,
            quantize_config=None,
            use_triton=False,
        )

        p = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            temperature=0,
            top_p=0.2,
            repetition_penalty=1.15,
        )

        hf_pipeline = HuggingFacePipeline(pipeline=p)
        return hf_pipeline


class RedPajamaIncite(AbstractLLMModel):
    def __init__(
        self, cache_folder, model_name="togethercomputer/RedPajama-INCITE-7B-Instruct", load_8bit=False, device="auto"
    ) -> None:
        super().__init__(
            cache_folder=cache_folder,
            model_name=model_name,
            prompt_template=REDPAJAMA_TEMPLATE,
            load_8bit=load_8bit,
        )
        self.device = device

    @property
    def model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            cache_dir=self.cache_folder,
            device_map="auto",
            load_in_8bit=self.load_8bit,
        )

        generate_text = transformers.pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            # device=device,
            device_map="auto",
            do_sample=True,
            temperature=0.01,
            max_new_tokens=128,
            top_p=0.2,
            model_kwargs={"cache_dir": self.cache_folder},
        )

        hf_pipeline = HuggingFacePipeline(pipeline=generate_text)
        return hf_pipeline


class LlamaCPPModel(AbstractLLMModel):
    def __init__(
        self,
        model_name: str,
        cache_folder: Path,
        model_path: Union[Path, str],
        prompt_template: str,
        llama_kwargs: dict,
        load_8bit: Optional[bool] = False,

    ) -> None:
        super().__init__(model_name, cache_folder, prompt_template, load_8bit)
        self.model_path = model_path
        self.llama_kwargs = llama_kwargs

    @property
    def model(self):
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        llm = LlamaCpp(
            model_path=self.model_path, callback_manager=callback_manager, verbose=True, **self.llama_kwargs
        )
        return llm
