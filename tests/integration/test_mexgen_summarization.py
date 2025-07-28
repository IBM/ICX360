import numpy as np
import pandas as pd
import pytest
import torch
from datasets import load_dataset  # for XSum dataset

from icx360.algorithms.mexgen import CLIME, LSHAP
from icx360.metrics import PerturbCurveEvaluator
from icx360.utils.general_utils import select_device
from icx360.utils.model_wrappers import GeneratedOutput

# UNCOMMENT THE "vllm" LINE BELOW IF A VLLM MODEL IS ADDED TO THE MODEL CATALOG IN ../conftest.py
MODEL_IDS = [
    "tiny-granite",
    pytest.param("distilbart", marks=pytest.mark.slow),
    pytest.param("granite-hf", marks=pytest.mark.slow),
    pytest.param("vllm", marks=(pytest.mark.slow, pytest.mark.vllm)),
]

# Create wrapped models and associated generation parameters
@pytest.fixture(scope="module", params=MODEL_IDS)
def wrapped_model_and_params(create_wrapped_model, model_catalog, request):
    wrapped_model = create_wrapped_model(**model_catalog[request.param])

    model_params = {}
    if request.param == "vllm":
        model_params["max_tokens"] = 100
        model_params["seed"] = 20250430
    elif request.param != "tiny-granite":
        model_params["max_new_tokens"] = 100

    if request.param in ("granite-hf", "vllm"):
        model_params["chat_template"] = True
        model_params["system_prompt"] = "Summarize the following article in one sentence. Do not preface the summary with anything."

    return wrapped_model, model_params, request.param

# Create explainers
@pytest.fixture(scope="module", params=["clime", "lshap"])
def explainer(wrapped_model_and_params, request):
    wrapped_model = wrapped_model_and_params[0]
    model_id = wrapped_model_and_params[2]

    if model_id == "tiny-granite":
        # Tiny models for text-only scalarizers
        # NEED A TINY NLI MODEL TO TEST NLI SCALARIZER
        model_bert_name = "google/bert_uncased_L-2_H-128_A-2"
        model_summ_name = "hf-internal-testing/tiny-random-BartForConditionalGeneration"
        explainer_kwargs = {
            # BARTScore with tiny-random-Bart yields an index out of range error that goes away with a larger BART model
            "sim_scores": ["bert", "st", "summ"],#, "bart"],
            "model_bert": model_bert_name,
            "model_summ": model_summ_name,
            # "model_bart": model_summ_name,
            "device": torch.device("cpu"),
        }
    else:
        # Larger models for text-only scalarizers
        model_nli_name = "microsoft/deberta-v2-xxlarge-mnli"
        model_summ_name = "sshleifer/distilbart-xsum-12-6"
        explainer_kwargs = {
            "model_nli": model_nli_name,
            "model_bert": model_nli_name,
            "model_summ": model_summ_name,
            "model_bart": model_summ_name,
            "device": select_device(),
        }

    if request.param == "clime":
        return CLIME(wrapped_model, scalarizer="text", **explainer_kwargs)
    elif request.param == "lshap":
        return LSHAP(wrapped_model, scalarizer="text", **explainer_kwargs)

# Document to summarize
@pytest.fixture(scope="module")
def document():
    # Zara/Inditex example from test split of XSum dataset
    dataset = load_dataset("xsum", split="test", trust_remote_code=True)
    document = dataset["document"][73]

    return document

# Sentence-level explanation
@pytest.fixture(scope="module")
def sent_level_explanation(explainer, document, wrapped_model_and_params):
    # Default parameters mostly suffice for sentence-level explanation
    model_params = wrapped_model_and_params[1]

    # Call explainer
    output_dict_sent = explainer.explain_instance(document, model_params=model_params)

    return output_dict_sent

# Test sentence-level explanation
def test_sent_level_explanation(sent_level_explanation, explainer, wrapped_model_and_params):
    output_dict_sent = sent_level_explanation
    score_labels = explainer.scalarized_model.sim_scores
    model_id = wrapped_model_and_params[2]

    assert isinstance(output_dict_sent, dict)

    # Check "output_orig" item
    assert "output_orig" in output_dict_sent
    assert isinstance(output_dict_sent["output_orig"], GeneratedOutput)
    assert isinstance(output_dict_sent["output_orig"].output_text, list)
    assert len(output_dict_sent["output_orig"].output_text) == 1
    assert isinstance(output_dict_sent["output_orig"].output_text[0], str)

    # Check "attributions" item
    assert "attributions" in output_dict_sent
    assert isinstance(output_dict_sent["attributions"], dict)
    attrib_df = pd.DataFrame(output_dict_sent["attributions"]).set_index("units")
    attrib_df = attrib_df[["unit_types"] + score_labels]
    assert len(attrib_df) >= 17

    # Check items under "attributions"
    assert isinstance(attrib_df.index[0], str)
    assert attrib_df["unit_types"].isin(["n", "s"]).all()
    for score_label in score_labels:
        assert np.issubdtype(attrib_df[score_label].dtype, np.floating)

def test_mixed_level_explanation(explainer, sent_level_explanation, wrapped_model_and_params):
    output_dict_sent = sent_level_explanation
    score_labels = explainer.scalarized_model.sim_scores

    # Parameters for mixed-level explanation
    units = output_dict_sent["attributions"]["units"]
    unit_types = output_dict_sent["attributions"]["unit_types"]
    ind_segment = np.zeros_like(units, dtype=bool)
    ind_segment[:2] = True
    segment_type = "ph"
    model_params = wrapped_model_and_params[1]

    # Call explainer
    output_dict_mixed = explainer.explain_instance(units,
                                                   unit_types,
                                                   ind_segment=ind_segment,
                                                   segment_type=segment_type,
                                                   model_params=model_params)

    # Check "output_orig" item
    assert len(output_dict_mixed["output_orig"].output_text) == 1
    assert isinstance(output_dict_mixed["output_orig"].output_text[0], str)

    # Check items under "attributions"
    attrib_df = pd.DataFrame(output_dict_mixed["attributions"]).set_index("units")
    attrib_df = attrib_df[["unit_types"] + score_labels]
    assert len(attrib_df) >= 28
    assert isinstance(attrib_df.index[0], str)
    for score_label in score_labels:
        assert np.issubdtype(attrib_df[score_label].dtype, np.floating)

def test_perturb_curve_evaluator(wrapped_model_and_params, sent_level_explanation, explainer):
    wrapped_model, model_params = wrapped_model_and_params[:2]
    output_dict_sent = sent_level_explanation
    score_labels = explainer.scalarized_model.sim_scores

    # Instantiate perturbation curve evaluator
    evaluator = PerturbCurveEvaluator(wrapped_model, scalarizer="prob")

    for score_label in score_labels:
        perturb_curve_sent = evaluator.eval_perturb_curve(output_dict_sent,
                                                          score_label,
                                                          token_frac=True,
                                                          model_params=model_params)

        assert isinstance(perturb_curve_sent, dict)
        assert "frac" in perturb_curve_sent
        assert isinstance(perturb_curve_sent["frac"], torch.Tensor)
        assert "prob" in perturb_curve_sent
        assert isinstance(perturb_curve_sent["prob"], torch.Tensor)
        assert len(perturb_curve_sent["frac"]) == len(perturb_curve_sent["prob"])
        # Check that perturbation curve has positive area
        assert (perturb_curve_sent["prob"][0] - perturb_curve_sent["prob"]).sum() > 0
