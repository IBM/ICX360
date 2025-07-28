import pytest
import torch

from icx360.utils.model_wrappers import GeneratedOutput
from icx360.utils.scalarizers import TextScalarizedModel

MODEL_IDS = ["tiny-granite", pytest.param("vllm", marks=(pytest.mark.slow, pytest.mark.vllm))]

@pytest.fixture(scope="module", params=MODEL_IDS)
def wrapped_model(create_wrapped_model, model_catalog, request):
    return create_wrapped_model(**model_catalog[request.param])

def test_scalarize_output(wrapped_model, tiny_model_prompts):
    # Create scalarized model
    # NEED A TINY NLI MODEL TO TEST NLI SCALARIZER
    # BARTScore with tiny-random-Bart yields an index out of range error that goes away with a larger BART model
    model_bert_name = "google/bert_uncased_L-2_H-128_A-2"
    model_summ_name = "hf-internal-testing/tiny-random-BartForConditionalGeneration"
    scalarized_model = TextScalarizedModel(wrapped_model,
                                           sim_scores=["bert", "st", "summ"],
                                           model_bert=model_bert_name,
                                           model_summ=model_summ_name,
                                           model_bart=model_summ_name,
                                           device=torch.device("cpu"))

    # Reference output for scalarization
    ref_output = GeneratedOutput(output_text=["Hello! How can I help you?"])

    # Compute log probability of reference output given prompts
    scores = scalarized_model.scalarize_output(inputs=tiny_model_prompts, ref_output=ref_output)

    assert isinstance(scores, dict)
    for score_label in scalarized_model.sim_scores:
        assert score_label in scores
        assert isinstance(scores[score_label], torch.Tensor)
        assert scores[score_label].shape[0] == len(tiny_model_prompts)
