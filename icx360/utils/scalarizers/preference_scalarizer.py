"""File containing class PreferenceScalarizer

This class is used to scalarize text using a preference model to measure the
change in preference for a contrastive response to the original prompt

"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from icx360.utils.scalarizers import Scalarizer


class PreferenceScalarizer(Scalarizer):
    """PreferenceScalarizer object.

    Instances of PreferenceScalarizer can call scalarize_output to produce
    scalarized version of input text accoring to change in preference of a
    contrastive response relative to the initial response

    Attributes
        _model: model for computing preference score
        _tokenizer: tokenizer of preference model
        _device: device on which to perform computations
    """

    def __init__(self, model_path='stanfordnlp/SteamSHP-flan-t5-large', device='cuda'):
        """Initialize preference scalarizer object.

            Args:
                model_path (str): preference model
                device (str): device on which to perform computations
        """

        super().__init__()
        self._tokenizer = T5Tokenizer.from_pretrained(model_path, device_map=device)
        self._model = T5ForConditionalGeneration.from_pretrained(model_path, device_map=device)
        self._device = device

    def scalarize_output(self, inputs, outputs, ref_input='', ref_output='', input_label=0, info=False):
        """Convert text input and outputs to numerical score

            Use preference score to scalarize. Compute preference for prompt
            inputs relative to two different responses, outputs and ref_output.

            Args:
                inputs (str): input prompt
                outputs (str): response to input prompt
                ref_input (str): placeholder. not used here.
                ref_output (str): response to contrastive prompt
                input_label (int): placeholder. not used here.
                info (bool): placeholder. not used here.

            Returns:
                score (float): scalarized output
                label_contrast (int): placeholder. not used here.
        """

        if outputs != ref_output: # only check preference if the responses are different
            # run preference model with two outputs
            input_text_preference = 'POST: ' + inputs + '\n\n RESPONSE A: ' + outputs + '. \n\n RESPONSE B: ' + ref_output + '. \n\n Which response is better? RESPONSE'
            # remove special tokens that are due to model being explained
            input_text_preference = input_text_preference.replace('<pad>','')
            input_text_preference = input_text_preference.replace('</s>','')
            x = self._tokenizer([input_text_preference], return_tensors='pt').input_ids.to(self._device)
            y = self._model.generate(x, return_dict_in_generate=True, output_scores=True, max_new_tokens=1)

            score_a = torch.exp(y.scores[0][:, 71]) / torch.exp(y.scores[0][:,:]).sum(axis=1).item()
            score_b = torch.exp(y.scores[0][:, 272]) / torch.exp(y.scores[0][:,:]).sum(axis=1).item()
            score = (score_a-score_b).item() # measure the difference in preference for a response generated from changing the prompt
        else:
            score = 0.0 # assume no preference between two equivalent responses
        label_contrast = 0

        return (score, label_contrast)
