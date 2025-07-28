import pytest
import torch
from openai import OpenAI
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BartForConditionalGeneration,
    BartTokenizerFast,
    T5ForConditionalGeneration,
    T5TokenizerFast,
)

from icx360.utils.general_utils import select_device
from icx360.utils.model_wrappers import HFModel, VLLMModel

### MODEL CATALOG

@pytest.fixture(scope="package")
def model_catalog():

    catalog = {
        # Tiny HuggingFace model for non-slow tests
        "tiny-granite": {
            "model_type": "hf",
            "args": {
                "model_name": "hf-internal-testing/tiny-random-GraniteForCausalLM",
                "tokenizer_kwargs": {"add_prefix_space": True},
            }
        },
        # Small "slow" model for summarization
        "distilbart": {
            "model_type": "hf",
            "args": {
                "model_name": "sshleifer/distilbart-xsum-12-6",
                "tokenizer_kwargs": {"add_prefix_space": True},
            }
        },
        # "Slow" encoder-decoder model for question answering
        "flan-t5": {
            "model_type": "hf",
            "args": {
                "model_name": "google/flan-t5-large",
            }
        },
        # "Slow" decoder-only model for summarization and question answering
        "granite-hf": {
            "model_type": "hf",
            "args": {
                "model_name": "ibm-granite/granite-3.3-2b-instruct",
                "model_kwargs": {"torch_dtype": torch.bfloat16},
                "tokenizer_kwargs": {"add_prefix_space": True},
            }
        },
        # "Slow" VLLM model for summarization and question answering
        "vllm": {
            "model_type": "vllm",
            "args": {
                # IF YOU HAVE A VLLM MODEL TO TEST, UNCOMMENT AND REPLACE THE FOLLOWING LINES WITH YOUR MODEL'S PARAMETERS
                # "model_name": "YOUR/MODEL-NAME",
                # "base_url": "https://YOUR/MODEL/URL",
                # "api_key": YOUR_API_KEY,
                # "openai_kwargs": DICTIONARY OF YOUR OPENAI KWARGS, OR LEAVE COMMENTED IF NONE
                # "tokenizer_kwargs": DICTIONARY OF YOUR TOKENIZER KWARGS, OR LEAVE COMMENTED IF NONE
            }
        },
    }

    return catalog

### WRAPPED MODEL FACTORIES

@pytest.fixture(scope="package")
def create_wrapped_hf_model():
    def _create_wrapped_hf_model(model_name, model_kwargs={}, tokenizer_kwargs={}):
        if "distilbart" in model_name:
            model_class, tokenizer_class = BartForConditionalGeneration, BartTokenizerFast
        elif "flan-t5" in model_name:
            model_class, tokenizer_class = T5ForConditionalGeneration, T5TokenizerFast
        else:
            model_class, tokenizer_class = AutoModelForCausalLM, AutoTokenizer

        model = model_class.from_pretrained(model_name, **model_kwargs)
        if "tiny" not in model_name:
            model = model.to(select_device())
        tokenizer = tokenizer_class.from_pretrained(model_name, **tokenizer_kwargs)

        return HFModel(model, tokenizer)

    return _create_wrapped_hf_model

@pytest.fixture(scope="package")
def create_wrapped_vllm_model():
    def _create_wrapped_vllm_model(model_name, base_url, api_key, openai_kwargs={}, tokenizer_kwargs={}):
        model = OpenAI(base_url=base_url, api_key=api_key, **openai_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

        return VLLMModel(model, model_name, tokenizer)

    return _create_wrapped_vllm_model

@pytest.fixture(scope="package")
def create_wrapped_model(create_wrapped_hf_model, create_wrapped_vllm_model):
    def _create_wrapped_model(model_type, args):
        if model_type == "hf":
            return create_wrapped_hf_model(**args)
        elif model_type == "vllm":
            return create_wrapped_vllm_model(**args)

    return _create_wrapped_model

### PROMPTS TO TINY MODEL
@pytest.fixture(scope="package")
def tiny_model_prompts():
    PROMPTS = ["Hello tiny model", "Goodbye tiny model"]

    return PROMPTS
