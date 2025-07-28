import pytest
import torch

from icx360.utils.scalarizers import BleuScalarizer


def test_scalarize_output():
    # Create scalarized model
    scalarized_model = BleuScalarizer(device='cpu')

    # References for scalarization
    inputs = "Hello! How can I help you?"
    outputs = "Get me a glass of water."
    ref_input = "How are you feeling?"
    ref_output = "I do not know."

    # Compute preference score
    (score, label_contrast) = scalarized_model.scalarize_output(inputs=inputs, outputs=outputs, ref_input=ref_input, ref_output=ref_output)

    assert isinstance(score, float)
    assert label_contrast == 0
