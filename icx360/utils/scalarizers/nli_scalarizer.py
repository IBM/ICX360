"""File containing class NLIScalarizer

This class is used to scalarize text using an Natural Language Inference (NLI)
score to measure the change in scores

"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from icx360.utils.scalarizers import Scalarizer


class NLIScalarizer(Scalarizer):
    """NLIScalarizer object.

    Instances of NLIScalarizer can call scalarize_output to produce
    scalarized version of input text accoring to change in NLI score

    Attributes
        _model: NLI model for computing contradiction score
        _tokenizer: tokenizer of NLI model
        _device: device on which to perform computations
    """

    def __init__(self, model_path='cross-encoder/nli-deberta-v3-base', device='cuda'):
        """Initialize nli scalarizer object.

            Args:
                model_path (str): NLI model for computing nli score
                device (str): device on which to perform computations
        """

        super().__init__()
        self._model = AutoModelForSequenceClassification.from_pretrained(model_path, device_map=device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._device = device

    def scalarize_output(self, inputs, outputs, ref_input='', ref_output='', input_label=0, info=False):
        """Convert text input and outputs to numerical score

            Use NLI score to scalarize. Compute score of NLI prediction of
            NLI(inputs, outputs) and compute change in score for that class of
            NLI(inputs, ref_output)

            Args:
                inputs (str): input prompt
                outputs (str): response to input prompt
                ref_input (str): placeholder. not used here.
                ref_output (str): response to contrastive prompt
                input_label (int): placeholder. not used here.
                info (bool): print extra information if True

            Returns:
                score (float): scalarized output
                label_contrast (int): placeholder. not used here.
        """

        # run nli model with two outputs
        features_nli = self._tokenizer([inputs, inputs],[outputs, ref_output],  padding=True, truncation=True, return_tensors="pt").to(self._device)
        self._model.eval()

        with torch.no_grad():
            scores_nli = self._model(**features_nli).logits # score computations not done on cuda
        if info == False:
            scores_np = torch.nn.functional.softmax(scores_nli, dim=1)
            lab = torch.argmax(scores_np[0,:]).item()
            score = (scores_np[0,lab] - scores_np[1,lab]).item() # change in class
            label_contrast = 0
        else:
            label_mapping = ['contradiction', 'entailment', 'neutral']
            labels = [label_mapping[score_max.item()] for score_max in scores_nli.argmax(dim=1)]
            print('NLI initial prediction: ' + labels[0])
            print('NLI modified prediction: ' + labels[1])
            score = 0
            label_contrast = 0

        return (score, label_contrast)
