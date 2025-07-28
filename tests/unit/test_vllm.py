import pytest

from icx360.utils.model_wrappers import GeneratedOutput

MODEL_ID = "vllm"

@pytest.fixture(scope="module")
def wrapped_vllm_model(create_wrapped_vllm_model, model_catalog):
    return create_wrapped_vllm_model(**model_catalog[MODEL_ID]["args"])

@pytest.mark.slow
@pytest.mark.vllm
def test_convert_input(wrapped_vllm_model, tiny_model_prompts):
    input_converted = wrapped_vllm_model.convert_input(tiny_model_prompts)

    assert isinstance(input_converted, list)
    assert len(input_converted) == len(tiny_model_prompts)

@pytest.mark.slow
@pytest.mark.vllm
@pytest.mark.parametrize("text_only", [True, False])
def test_generate(wrapped_vllm_model, tiny_model_prompts, text_only):
    output_obj = wrapped_vllm_model.generate(tiny_model_prompts, text_only=text_only)

    if text_only:
        assert isinstance(output_obj, list)
        assert len(output_obj) == len(tiny_model_prompts)
    else:
        assert isinstance(output_obj, GeneratedOutput)
        assert isinstance(output_obj.output_text, list)
        assert len(output_obj.output_text) == len(tiny_model_prompts)
