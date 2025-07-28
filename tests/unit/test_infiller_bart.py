import pytest
import torch

from icx360.utils.infillers import BART_infiller


@pytest.fixture(scope="module")
def infiller():
    # Create infiller
    _infiller = BART_infiller.BART_infiller(model_path='facebook/bart-base', device=torch.device("cpu"))

    return _infiller

def test_encode(infiller):
    text = "Please get me my " + infiller.mask_string + " from the basement."
    ans = [0, 6715, 120, 162, 127, 50264, 31, 5, 12288, 4, 2]
    s = infiller.encode(text, add_special_tokens=True)

    assert isinstance(s, list)
    for i in range(len(s)):
        assert s[i] == ans[i]

def test_decode(infiller):
    tokens = [0, 6715, 120, 162, 127, 50264, 31, 5, 12288, 4, 2]

    text = infiller.decode(tokens)
    ans = "Please get me my from the basement."

    assert text == ans

def test_generate(infiller):
    tokens = [0, 6715, 120, 162, 127, 50264, 31, 5, 12288, 4, 2]

    (generated_ids, mask_filled) = infiller.generate(tokens, masked_word='bag', num_return_sequences=3, return_mask_filled=True)

    assert isinstance(generated_ids, list)
    # check that each token appears in generation unless it is the mask token
    for i in tokens:
        assert i in generated_ids or i==infiller.mask_string_encoded
    assert len(generated_ids) >= len(tokens)

def test_get_infilled_mask(infiller):
    x_enc = [0, 6715, 120, 162, 127, 50264, 31, 5, 12288, 4, 2]
    y_enc = [0, 6715, 120, 162, 127, 2682, 31, 5, 12288, 4, 2]

    (mask_filled, inds_infill) = infiller.get_infilled_mask(x_enc, y_enc, return_tokens=True)

    assert mask_filled == "stuff"
    assert isinstance(inds_infill, list)
    assert(len(inds_infill) == 1)
    assert inds_infill[0] == 2682

def test_similar(infiller):
    word = "bag"
    fill_in1 = "Please get me my bag from the basement."
    fill_in2 = "Please get me my jacket from the basement."

    ret1 = infiller.similar(word, fill_in1)
    ret2 = infiller.similar(word, fill_in2)

    assert ret1
    assert not ret2
