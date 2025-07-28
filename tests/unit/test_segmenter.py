import pytest

from icx360.utils.segmenters import SpaCySegmenter, exclude_non_alphanumeric


# Segmenter
@pytest.fixture(scope="module")
def segmenter():
    # Use smallest spaCy model for testing
    return SpaCySegmenter("en_core_web_sm")

# Paragraph-level inputs
@pytest.fixture(scope="module")
def inputs_para():
    units_para = [
        "unit to ignore",
        "I love the cold of Boston. However, I would prefer the heat of Rio.",
        "\n",
        "another unit to ignore because of type 'n', despite ind_segment==True",
    ]
    ind_segment_para = [False, True, True, True]
    unit_types_para = ["p", "p", "p", "n"]

    return {"input_text": units_para, "ind_segment": ind_segment_para, "unit_types": unit_types_para}

# Reference outputs
@pytest.fixture(scope="module")
def ref_outputs_sent():
    units_sent_ref = [
        "unit to ignore",
        "I love the cold of Boston. ",
        "However, I would prefer the heat of Rio.",
        "\n",
        "another unit to ignore because of type 'n', despite ind_segment==True",
    ]
    unit_types_sent_ref = ["p", "s", "s", "s", "n"]

    return units_sent_ref, unit_types_sent_ref

@pytest.fixture(scope="module")
def ref_outputs_word():
    units_word_ref = [
        "unit to ignore",
        "I ", "love ", "the ", "cold ", "of ", "Boston", ". ",
        "However", ", ", "I ", "would ", "prefer ", "the ", "heat ", "of ", "Rio", ".",
        "\n",
        "another unit to ignore because of type 'n', despite ind_segment==True",
    ]
    unit_types_word_ref = ["p"] + ["w"] * 6 + ["n"] + ["w", "n"] + ["w"] * 7 + ["n"] * 3

    return units_word_ref, unit_types_word_ref

@pytest.fixture(scope="module")
def ref_outputs_phrase():
    units_phrase_ref = [
        "unit to ignore",
        "I ", "love ", "the cold of Boston", ". ",
        "However", ", ", "I ", "would ", "prefer ", "the heat of Rio", ".",
        "\n",
        "another unit to ignore because of type 'n', despite ind_segment==True",
    ]

    return units_phrase_ref

# Test paragraph-level segmentation
def test_segment_para(segmenter, inputs_para):
    # segment_type="p" should not segment at all
    units_sent, unit_types_sent, _ = segmenter.segment_units(**inputs_para, segment_type="p")
    assert units_sent == inputs_para["input_text"]
    assert unit_types_sent == inputs_para["unit_types"]

# Sentence-level outputs
@pytest.fixture(scope="module")
def outputs_sent(segmenter, inputs_para):
    units_sent, unit_types_sent, _ = segmenter.segment_units(**inputs_para, segment_type="s")
    ind_segment_sent = [False, True, True, True, True]

    return {"input_text": units_sent, "ind_segment": ind_segment_sent, "unit_types": unit_types_sent}

# Test sentence-level, word-level, and phrase-level segmentation
def test_segment_sent(outputs_sent, ref_outputs_sent):
    units_sent, unit_types_sent = outputs_sent["input_text"], outputs_sent["unit_types"]
    units_sent_ref, unit_types_sent_ref = ref_outputs_sent

    assert units_sent == units_sent_ref
    assert unit_types_sent == unit_types_sent_ref

def test_segment_word(segmenter, outputs_sent, ref_outputs_word):
    units_word, unit_types_word, _ = segmenter.segment_units(**outputs_sent, segment_type="w")
    unit_types_word = exclude_non_alphanumeric(unit_types_word, units_word)
    units_word_ref, unit_types_word_ref = ref_outputs_word

    assert units_word == units_word_ref
    assert unit_types_word == unit_types_word_ref

def test_segment_phrase(segmenter, outputs_sent, ref_outputs_phrase):
    units_phrase, unit_types_phrase, _ = segmenter.segment_units(**outputs_sent, segment_type="ph")
    unit_types_phrase = exclude_non_alphanumeric(unit_types_phrase, units_phrase)

    assert units_phrase == ref_outputs_phrase
    assert len(unit_types_phrase) == 14
    assert unit_types_phrase[4] == "n"
