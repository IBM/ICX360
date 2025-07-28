import pytest
import torch

from icx360.utils.model_wrappers import GeneratedOutput

MODEL_ID = "tiny-granite"

@pytest.fixture(scope="module")
def wrapped_hf_model(create_wrapped_hf_model, model_catalog):
    return create_wrapped_hf_model(**model_catalog[MODEL_ID]["args"])

def test_convert_input(wrapped_hf_model, tiny_model_prompts):
    input_encoding = wrapped_hf_model.convert_input(tiny_model_prompts)

    assert "input_ids" in input_encoding
    assert isinstance(input_encoding["input_ids"], torch.LongTensor)
    assert input_encoding["input_ids"].shape[0] == len(tiny_model_prompts)

@pytest.mark.parametrize("text_only", [True, False])
def test_generate(wrapped_hf_model, tiny_model_prompts, text_only):
    output_obj = wrapped_hf_model.generate(tiny_model_prompts, text_only=text_only)

    if text_only:
        assert isinstance(output_obj, list)
        assert len(output_obj) == len(tiny_model_prompts)
    else:
        assert isinstance(output_obj, GeneratedOutput)
        assert isinstance(output_obj.output_text, list)
        assert isinstance(output_obj.output_ids, torch.LongTensor)
        assert len(output_obj.output_text) == len(tiny_model_prompts)
        assert output_obj.output_ids.shape[0] == len(tiny_model_prompts)
