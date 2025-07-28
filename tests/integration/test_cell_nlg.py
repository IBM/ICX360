import pytest

from icx360.algorithms.cell.CELL import CELL
from icx360.algorithms.cell.mCELL import mCELL
from icx360.utils.general_utils import select_device

# uncomment the vllm pytest below if added in ../conftest.py
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
        model_params["system_prompt"] = "Please respond to the following statement or question very briefly in less than 10 words."

    return wrapped_model, model_params, request.param


@pytest.fixture(scope="module", params=["t5", "bart"])
def infiller(request):
    return request.param

@pytest.fixture(scope="module", params=["preference", "nli", "contradiction", "bleu"])
def scalarizer(request):
    return request.param

# Create explainers
@pytest.fixture(scope="module", params=["cell", "mcell"])
def explainer(wrapped_model_and_params, infiller, scalarizer, request):
    device = select_device()
    wrapped_model = wrapped_model_and_params[0]
    if request.param == "cell":
        return CELL(wrapped_model, num_return_sequences=4, infiller=infiller, scalarizer=scalarizer, device=device)
    elif request.param == "mcell":
        return mCELL(wrapped_model, num_return_sequences=4, infiller=infiller, scalarizer=scalarizer, device=device)

@pytest.fixture(scope="module")
def prompt():
    prompt = "What are the most popular activities for children for elementary school age?"

    return prompt

# Test paragraph-level explanation
def test_explanation(prompt, explainer):

    if isinstance(explainer, CELL):
        result = explainer.explain_instance(prompt, radius=3, budget=10, split_k=2, epsilon_contrastive=0.25)
    elif isinstance(explainer, mCELL):
        result = explainer.explain_instance(prompt, split_k=2, epsilon_contrastive=0.25)
    else:
        assert 0, 'Invalid explainer object'

    assert isinstance(result, dict)
    assert isinstance(result['prompt_cell'], str)
    assert isinstance(result['response_cell'], str)
    assert isinstance(result['output_original'], str)
    assert isinstance(result['tokens_cell'], list)
    assert isinstance(result['mask_order'], list)
