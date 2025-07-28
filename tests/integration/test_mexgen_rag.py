import json
import os

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
        model_params["system_prompt"] = "You are an AI assistant that provides a *very short* answer to the user *solely based on the provided documents*, aiming to answer the user as correctly as possible. If no documents are provided, answer from your memory."

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
def create_template_for_flant5_generation(documents, question):

    system_template = "Answer the user Question using the following Context: \n\n"

    templated_system_prompt = system_template

    DOCUMENTS_template = "Context:\n\n"
    QUESTION_template = "\n\nQuestion: "

    templated = [templated_system_prompt + DOCUMENTS_template]
    documents = [doc + "\n\n" for doc in documents]

    templated.extend(documents)
    templated.append(QUESTION_template + question)

    unit_types = ["n"] + len(documents) * ["p"] + ["n"]

    return templated, unit_types

def create_template_for_granite_generation(documents, question):


    DOCUMENTS_template = "Documents: \n"
    QUESTION_template = "\n\nQuestion: "

    templated = [DOCUMENTS_template]
    documents = [f"Document {idx}: {doc} \n" for idx, doc in enumerate(documents)]

    templated.extend(documents)
    templated.append(QUESTION_template + question)

    unit_types = ["n"] + len(documents) * ["p"] + ["n"]

    return templated, unit_types

@pytest.fixture(scope="module")
def rag_prompt(wrapped_model_and_params, ):
    question = "What is the coldest state of the contiguous USA?"

    # Load retrieved documents from file instead of re-retrieving to avoid dependency on faiss package
    path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(path, "..", "..", "examples", "mexgen", "RAG_retrieved_docs.json"), "r") as f:
        documents = json.load(f)

    model_id = wrapped_model_and_params[2]
    if model_id == "flan-t5":
        prompt, unit_types = create_template_for_flant5_generation(documents, question)
    else:
        prompt, unit_types = create_template_for_granite_generation(documents, question)

    return prompt, unit_types

# Paragraph-level explanation
@pytest.fixture(scope="module")
def para_level_explanation(explainer, rag_prompt, wrapped_model_and_params):
    # Parameters for paragraph-level explanation
    units, unit_types = rag_prompt
    ind_segment = [False] * len(units)
    segment_type = "p"
    model_params = wrapped_model_and_params[1]

    # Call explainer
    output_dict_para = explainer.explain_instance(units,
                                                  unit_types,
                                                  ind_segment=ind_segment,
                                                  segment_type=segment_type,
                                                  model_params=model_params)

    return output_dict_para

# Test paragraph-level explanation
def test_para_level_explanation(para_level_explanation, rag_prompt, wrapped_model_and_params):
    output_dict_para = para_level_explanation
    units, unit_types = rag_prompt
    model_id = wrapped_model_and_params[2]

    assert isinstance(output_dict_para, dict)

    # Check "output_orig" item
    assert "output_orig" in output_dict_para
    assert isinstance(output_dict_para["output_orig"], GeneratedOutput)
    assert isinstance(output_dict_para["output_orig"].output_text, list)
    assert len(output_dict_para["output_orig"].output_text) == 1
    assert isinstance(output_dict_para["output_orig"].output_text[0], str)

    # Check "attributions" item
    assert "attributions" in output_dict_para
    assert isinstance(output_dict_para["attributions"], dict)
    attrib_df = pd.DataFrame(output_dict_para["attributions"])
    assert len(attrib_df) == len(units)

    # Check items under "attributions"
    assert isinstance(attrib_df.loc[0, "units"], str)
    assert (attrib_df["unit_types"] == unit_types).all()
    assert np.issubdtype(attrib_df["prob"].dtype, np.floating)

def test_mixed_level_explanation(explainer, para_level_explanation, wrapped_model_and_params):
    output_dict_para = para_level_explanation

    # Parameters for mixed-level explanation
    units = output_dict_para["attributions"]["units"]
    unit_types = output_dict_para["attributions"]["unit_types"]
    ind_segment = np.zeros_like(units, dtype=bool)
    ind_segment[2] = True
    segment_type = "s"
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
    assert attrib_df["unit_types"].isin(["n", "p", "s"]).all()
    assert np.issubdtype(attrib_df["prob"].dtype, np.floating)

def test_perturb_curve_evaluator(wrapped_model_and_params, para_level_explanation):
    wrapped_model, model_params = wrapped_model_and_params[:2]
    output_dict_para = para_level_explanation

    # Instantiate perturbation curve evaluator
    evaluator = PerturbCurveEvaluator(wrapped_model, scalarizer="prob")

    perturb_curve_para = evaluator.eval_perturb_curve(output_dict_para,
                                                      "prob",
                                                      token_frac=True,
                                                      model_params=model_params)

    assert isinstance(perturb_curve_para, dict)
    assert "frac" in perturb_curve_para
    assert isinstance(perturb_curve_para["frac"], torch.Tensor)
    assert "prob" in perturb_curve_para
    assert isinstance(perturb_curve_para["prob"], torch.Tensor)
    assert len(perturb_curve_para["frac"]) == len(perturb_curve_para["prob"])
    # Check that perturbation curve has positive area
    assert (perturb_curve_para["prob"][0] - perturb_curve_para["prob"]).sum() > 0
