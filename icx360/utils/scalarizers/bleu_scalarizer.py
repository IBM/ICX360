"""File containing class BleuScalarizer

This class is used to scalarize text using a Bleu metric

"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import evaluate

from icx360.utils.scalarizers import Scalarizer


class BleuScalarizer(Scalarizer):
    """BleuScalarizer object.

    Instances of BleuScalarizer can call scalarize_output to produce
    scalarized version of input text accoring to BLEU score

    Attributes
        _bleu: model for computing BLEU score
        _device: device on which to perform computations
    """

    def __init__(self, model_path='', device='cuda', experiment_id='id'):
        """Initialize bleu scalarizer object.

            Args:
                model_path (str): placeholder. deprecated here.
                device (str): device on which to perform computations
                experiment_id (str): unique string for parallel scores to be
                    computed without issue
        """

        super().__init__()
        self._bleu = evaluate.load("bleu", experiment_id=experiment_id)
        self._device = device

    def scalarize_output(self, inputs, outputs, ref_input='', ref_output='', input_label=0, info=False):
        """Convert text input and outputs to numerical score

            Use BLEU score to scalarize. Compute BLEU(outputs, ref_output) and
            BLEU(inputs, ref_input) and output a linear combination of BLEU
            scores

            Args:
                inputs (str): input prompt
                outputs (str): response to input prompt
                ref_input (str): contrastive prompt
                ref_output (str): response to contrastive prompt
                input_label (int): placeholder. not used here.
                info (bool): placeholder. not used here.

            Returns:
                score (float): scalarized output
                label_contrast (int): placeholder. not used here.
        """

        predictions = [outputs]
        references = [[ref_output]]
        results_response = self._bleu.compute(predictions=predictions, references=references)
        predictions = [inputs]
        references = [[ref_input]]
        results_prompt = self._bleu.compute(predictions=predictions, references=references)
        score_response = 1. - results_response['bleu'] # subtract from 1. because we want higher score to mean more dissimilar
        score_prompt = results_prompt['bleu'] # we want prompt to change as little as possible
        score = 0.8*score_response+0.2*score_prompt # compute weighting of both bleu scores
        label_contrast = 0

        return (score, label_contrast)
