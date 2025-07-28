import pytest
import torch

from icx360.utils.scalarizers import ContradictionScalarizer


def test_scalarize_output():
    # Create scalarized model
    scalarized_model = ContradictionScalarizer(device='cpu')

    # References for scalarization
    inputs = ''
    outputs = "Get me a glass of water."
    ref_output = "I do not know."

    # Compute contradiction score
    (score, label_contrast) = scalarized_model.scalarize_output(inputs=inputs, outputs=outputs, ref_output=ref_output)

    assert isinstance(score, float)
    assert label_contrast == 0

    # Test with info=True
    (score, label_contrast) = scalarized_model.scalarize_output(inputs=inputs, outputs=outputs, ref_output=ref_output, info=True)

    assert score == 0
    assert label_contrast == 0

def test_predict_contradiction():
    # Create scalarized model
    scalarized_model = ContradictionScalarizer(device='cpu')

    # References for scalarization
    inputs = ''
    outputs = "Get me a glass of water."
    ref_output = "I do not know."

    # Compute label
    label = scalarized_model.predict_contradiction(inputs=input, outputs=outputs, ref_output=ref_output)

    assert label in [0, 1]
