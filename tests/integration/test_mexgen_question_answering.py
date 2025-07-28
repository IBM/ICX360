import numpy as np
import pandas as pd
import pytest
import torch

from icx360.algorithms.mexgen import CLIME, LSHAP
from icx360.metrics import PerturbCurveEvaluator
from icx360.utils.model_wrappers import GeneratedOutput

# UNCOMMENT THE "vllm" LINE BELOW IF A VLLM MODEL IS ADDED TO THE MODEL CATALOG IN ../conftest.py
MODEL_IDS = [
    "tiny-granite",
    pytest.param("flan-t5", marks=pytest.mark.slow),
    pytest.param("granite-hf", marks=pytest.mark.slow),
    pytest.param("vllm", marks=(pytest.mark.slow, pytest.mark.vllm)),
]

# Create wrapped models and associated generation parameters
@pytest.fixture(scope="module", params=MODEL_IDS)
def wrapped_model_and_params(create_wrapped_model, model_catalog, request):
    wrapped_model = create_wrapped_model(**model_catalog[request.param])

    model_params = {}
    if request.param == "vllm":
        model_params["max_tokens"] = 64
        model_params["seed"] = 20250430
    elif request.param != "tiny-granite":
        model_params["max_new_tokens"] = 64

    if request.param in ("granite-hf", "vllm"):
        model_params["chat_template"] = True
        model_params["system_prompt"] = "Please answer the question using only the provided context."

    return wrapped_model, model_params, request.param

# Create explainers
@pytest.fixture(scope="module", params=["clime", "lshap"])
def explainer(wrapped_model_and_params, request):
    wrapped_model = wrapped_model_and_params[0]
    if request.param == "clime":
        return CLIME(wrapped_model, scalarizer="prob")
    elif request.param == "lshap":
        return LSHAP(wrapped_model, scalarizer="prob")

# Prepare prompt
@pytest.fixture(scope="module")
def qa_prompt():
    context = ('In July 2002, Beyonce continued her acting career playing Foxxy Cleopatra alongside Mike Myers in the'
            ' comedy film, Austin Powers in Goldmember, which spent its first weekend atop the US box office and grossed'
            ' $73 million. Beyonce released "Work It Out" as the lead single from its soundtrack album which entered'
            ' the top ten in the UK, Norway, and Belgium. In 2003, Beyonce starred opposite Cuba Gooding, Jr., in a'
            " musical comedy as Lilly, a single mother whom Gooding's character falls in love"
            ' with. The film received mixed reviews from critics but grossed $30 million in the U.S. Beyonce released'
            " \"Fighting Temptation\" as the lead single from the film's soundtrack album, with Missy Elliott, MC Lyte, and"
            " Free which was also used to promote the film. Another of Beyonce's contributions to the soundtrack,"
            ' "Summertime", fared better on the US charts.')

    question = "What movie did Beyonce star in in 2003?"
    prompt = ["Context: ", context, "\n\nQuestion: ", question, "\n\nAnswer: "]

    return prompt

# Sentence-level explanation
@pytest.fixture(scope="module")
def sent_level_explanation(explainer, qa_prompt, wrapped_model_and_params):
    # Parameters for sentence-level explanation
    unit_types = ["n", "p", "n", "n", "n"]
    ind_segment = [False, True, False, False, False]
    segment_type = "s"
    model_params = wrapped_model_and_params[1]

    # Call explainer
    output_dict_sent = explainer.explain_instance(qa_prompt,
                                                  unit_types,
                                                  ind_segment=ind_segment,
                                                  segment_type=segment_type,
                                                  model_params=model_params)

    return output_dict_sent

# Test sentence-level explanation
def test_sent_level_explanation(sent_level_explanation, wrapped_model_and_params):
    output_dict_sent = sent_level_explanation
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
    attrib_df = pd.DataFrame(output_dict_sent["attributions"])
    assert len(attrib_df) == 10

    # Check items under "attributions"
    assert isinstance(attrib_df.loc[0, "units"], str)
    assert (attrib_df["unit_types"] == ["n"] + ["s"] * 6 + ["n"] * 3).all()
    assert np.issubdtype(attrib_df["prob"].dtype, np.floating)
    if model_id != "tiny-granite":
        assert attrib_df["prob"].argmax() == 3

def test_mixed_level_explanation(explainer, sent_level_explanation, wrapped_model_and_params):
    output_dict_sent = sent_level_explanation

    # Parameters for mixed-level explanation
    units = output_dict_sent["attributions"]["units"]
    unit_types = output_dict_sent["attributions"]["unit_types"]
    ind_segment = np.zeros(10, dtype=bool)
    ind_segment[3] = True
    segment_type = "w"
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
    attrib_df = pd.DataFrame(output_dict_mixed["attributions"])
    assert isinstance(attrib_df.loc[0, "units"], str)
    assert attrib_df["unit_types"].isin(["n", "s", "w"]).all()
    assert np.issubdtype(attrib_df["prob"].dtype, np.floating)

def test_perturb_curve_evaluator(wrapped_model_and_params, sent_level_explanation):
    wrapped_model, model_params = wrapped_model_and_params[:2]
    output_dict_sent = sent_level_explanation

    # Instantiate perturbation curve evaluator
    evaluator = PerturbCurveEvaluator(wrapped_model, scalarizer="prob")

    perturb_curve_sent = evaluator.eval_perturb_curve(output_dict_sent,
                                                      "prob",
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
