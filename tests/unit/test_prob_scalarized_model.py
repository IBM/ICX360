import pytest
import torch

from icx360.utils.model_wrappers import GeneratedOutput, HFModel, VLLMModel
from icx360.utils.scalarizers import ProbScalarizedModel

MODEL_IDS = ["tiny-granite", pytest.param("vllm", marks=(pytest.mark.slow, pytest.mark.vllm))]

@pytest.fixture(scope="module", params=MODEL_IDS)
def wrapped_model(create_wrapped_model, model_catalog, request):
    return create_wrapped_model(**model_catalog[request.param])

def test_scalarize_output(wrapped_model, tiny_model_prompts):
    scalarized_model = ProbScalarizedModel(wrapped_model)

    # Reference output for scalarization
    if isinstance(wrapped_model, HFModel):
        ref_output = GeneratedOutput(output_ids=torch.LongTensor([[1, 2, 3]]))
    elif isinstance(wrapped_model, VLLMModel):
        ref_output = GeneratedOutput(output_text=["Hello! How can I help you?"])

    # Compute log probability of reference output given prompts
    probs = scalarized_model.scalarize_output(inputs=tiny_model_prompts, ref_output=ref_output)

    assert isinstance(probs, torch.Tensor)
    assert probs.shape[0] == len(tiny_model_prompts)
