from collections import namedtuple
import enum
from pathlib import Path
from typing import Any, List, Mapping, Optional, Union

import torch
import transformers
from dotenv import load_dotenv
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.llms.base import LLM
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          StoppingCriteria, StoppingCriteriaList, pipeline)

load_dotenv()



class ModelConfig(enum.Enum):
    OPENAI_GPT35 = "openai-gpt35"
    DOLLY3B = "databricks-dolly3b"
    DOLLY7B = 'databricks-dolly7b'
    MPT7B = 'mosaic-mpt7b-instruct'
    FALCON7B = 'falcon-7b-instruct'

# USed to group llm settings for the caller
LLMSettings = namedtuple("LLMSettins", "llm prompt")

def get_llm_model(model_name: str, cache_folder_root: Union[str, Path]):
    
    model = ModelConfig(model_name)
    
    
    prompt = None
    
    if model == ModelConfig.OPENAI_GPT35:
        llm = ChatOpenAI(temperature = 0.0)
        prompt = None
    
    elif model == ModelConfig.DOLLY3B:
        model_instance = LLMDatabricksDollyV2(cache_folder=cache_folder_root, model_name="databricks/dolly-v2-3b")
        llm = model_instance.model
        template = model_instance.prompt_template
        
        prompt =  PromptTemplate(
                        input_variables=["context", "question"],
                            template=template)
        
    
    elif model == ModelConfig.DOLLY7B:
        model_instance = LLMDatabricksDollyV2(cache_folder=cache_folder_root, model_name="databricks/dolly-v2-7b")

        llm = model_instance.model
        template = model_instance.prompt_template
        
        prompt =  PromptTemplate(
                        input_variables=["context", "question"],
                            template=template)
        

    elif model == ModelConfig.MPT7B:
        model_instance = LLMMosaicMPT(cache_folder=cache_folder_root, model_name="mosaicml/mpt-7b-instruct")
        
        llm = model_instance.model
        template = model_instance.prompt_template
        
        prompt =  PromptTemplate(
                        input_variables=["context", "question"],
                            template=template)
        
    elif model == ModelConfig.FALCON7B:
        llm = LLMFalcon(cache_folder=cache_folder_root, model_name="tiiuae/falcon-7b-instruct").model
    
    else:
        raise TypeError(f"Invalid model type. Got {model_name}")
    return LLMSettings(llm = llm, prompt=prompt)


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



class LLMDatabricksDollyV2:
    pipeline_kwargs = dict(
        torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", return_full_text=True
    )
    
    template = """### Instruction:
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

### Context: 
---------------
{context}
---------------

### Question: {question}
"""

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
    prompt_template = """### Instruction:
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

### Context: 
---------------
{context}
---------------

### Question: {question}
"""

    def __init__(self, cache_folder, model_name="mosaicml/mpt-7b-instruct", device = "auto") -> None:
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

        config = transformers.AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        
        config.init_device = device

        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name,
            config=config,
            torch_dtype=torch.bfloat16,
            cache_dir=self.cache_folder,
            trust_remote_code=True
        )
        

        
        tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", device = device)

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
            device=device,
            #device_map="auto",
            # we pass model parameters here too
            stopping_criteria=stopping_criteria,  # without this model will ramble
            temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
           # top_p=0.15,  # select from top tokens whose probability add up to 15%
           # top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
            max_new_tokens=512,  # mex number of tokens to generate in the output
           # repetition_penalty=1.1,  # without this output begins repeating
            model_kwargs={"cache_dir": self.cache_folder},
        )

        hf_pipeline = HuggingFacePipeline(pipeline=generate_text)
        return hf_pipeline


class LLMFalcon:


    def __init__(self, cache_folder, model_name="tiiuae/falcon-7b-instruct", device = "auto") -> None:
        self.cache_folder = cache_folder
        self.model_name = model_name
        self.device = device

    @property
    def model(self):
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model_kwargs = {'temperature':0.01, "cache_dir": self.cache_folder}
        
        pipeline = transformers.pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
                max_length=2000,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                model_kwargs=model_kwargs,
            )

        llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = model_kwargs)
        return llm