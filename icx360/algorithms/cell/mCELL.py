"""File containing class mCELL

mCELL is an explainer class that contains function explain_instance that
provides contrastive explanations of input instances. The algorithm for
providing explanations is described as m-Cell in: CELL your Model: Contrastive
Explanations for Large Language Models, Ronny Luss, Erik Miehling, Amit
Dhurandhar. https://arxiv.org/abs/2406.11785

"""

import re

import numpy as np
import torch

from icx360.algorithms.lbbe import LocalBBExplainer
from icx360.utils.infillers import BART_infiller, T5_infiller
from icx360.utils.scalarizers import (
    BleuScalarizer,
    ContradictionScalarizer,
    NLIScalarizer,
    PreferenceScalarizer,
)


class mCELL(LocalBBExplainer):
    """mCELL Explainer object.

    Instances of mCELL contain information about the LLM model being explained.
    These instances are used to explain LLM responses on input text using a
    myopic algorithm.

    Attributes:
        _model: model that we want to explain (based on
            icx360/utils/model_wrappers)
        _infiller: string for function used to input text with a mask token and
            output text with mask replaced by text
        _num_return_sequences: integer number of sequences returned when doing
            generation for mask infilling
        _scalarizer_name: string of scalarizer to use to determine if a
            contrast is found (must be from ['shp', 'nli', 'bleu']
        _scalarizer_type: string specifying either 'distance' for explaining
            LLM generation using distances or 'classifier' for explaining a
            classifier
        _scalarizer_func: function used to do scalarization from
            icx360/utils/scalarizers
        _generation: boolean specifying whether the model being explained
            performs true generation (as opposed to having output==input
            for classification)
        _device: string detailing device on which to perform all operations
            (must be from ['cpu', 'cuda', 'mps']). should be same as model
            being explained
    """

    def __init__(self, model, infiller='bart', num_return_sequences=1, scalarizer='shp', scalarizer_model_path=None, scalarizer_type='distance', generation=True, experiment_id='id', device=None):
        """Initialize contrastive explainer.

        Args:
            model: model that we want to explain (based on
                icx360/utils/model_wrappers)
            infiller (str): selects function used to input text with a mask
                token and output text with mask replaced by text
            num_return_sequences (int): number of sequences returned
                when doing generation for mask infilling
            scalarizer (str): select which scalarizer to use to determine
                if a contrast is found (must be from ['shp', 'nli',
                'bleu', 'implicit_hate', 'stigma'])
            scalarizer_model_path (str): allow user to pass a model path for
            scalarizers (e.g., choose 'stanfordnlp/SteamSHP-flan-t5-xl'
            instead of default 'stanfordnlp/SteamSHP-flan-t5-large')
            scalarizer_type (str): 'distance' for explaining LLM generation
                using distances, 'classifier' for explaining a classifier
            generation (bool): the model being explained performs true
                generation (as opposed to having output==input)
            experiment_id (str): passed to evaluate.load for certain
                scalarizers. This is used if several distributed evaluations
                share the same file system.
            device (str): device on which to perform all operations (must be
                from ['cpu', 'cuda', 'mps']). should be same as model being
                explained
        """

        self._model = model
        if device:
            self._device = device
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self._num_return_sequences = num_return_sequences

        if infiller == 'bart':
            self._infiller = BART_infiller.BART_infiller(device=device)
        elif infiller == 't5':
            self._infiller = T5_infiller.T5_infiller(device=device)
        else:
            raise Exception("mCELL received parameter value for infiller that is not recognized")

        self._scalarizer_name  = scalarizer
        self._scalarizer_type = scalarizer_type
        if scalarizer == 'preference':
            if scalarizer_model_path:
                self._scalarizer_func = PreferenceScalarizer(model_path=scalarizer_model_path, device=device)
            else:
                self._scalarizer_func = PreferenceScalarizer(device=device)
        elif scalarizer == 'nli':
            if scalarizer_model_path:
                self._scalarizer_func = NLIScalarizer(model_path=scalarizer_model_path, device=device)
            else:
                self._scalarizer_func = NLIScalarizer(device=device)
        elif scalarizer == 'contradiction':
            if scalarizer_model_path:
                self._scalarizer_func = ContradictionScalarizer(model_path=scalarizer_model_path, device=device)
            else:
                self._scalarizer_func = ContradictionScalarizer(device=device)
        elif scalarizer == 'bleu':
            self._scalarizer_func = BleuScalarizer(experiment_id=experiment_id)
        else:
            print('INVALID SCALARIZER')
        self._generation = generation

    def splitTextByK(self, str, k):
        """Split text into words.

        Args:
            str (str): string to be split
            k (int): number of consecutive words to keep together

        Returns:
            grouped_words (str list): list of words which when concatenated
                retrieves the input str
        """

        sentences_iter = re.finditer(r"[.!?;]", str)
        grouped_words = []
        start=0
        for sentence_iter in sentences_iter:
            end = sentence_iter.end()
            sentence = str[start:end].strip()
            words = sentence.split(' ')
            grouped_words.extend([' '.join(words[i: i + k]) for i in range(0, len(words), k)])
            start = end
        if start == 0: # special case for no punctuations found
            words = str.split(' ')
            grouped_words.extend([' '.join(words[i: i + k]) for i in range(0, len(words), k)])
        return grouped_words

    def explain_instance(self, input_text, epsilon_contrastive=.5, epsilon_iter=.001, split_k=1, no_change_max_iters=3, info=True, ir=False, model_params={}):
        """Provide explanations of large language model applied to prompt input_text

        Provide a contrastive explanation by changing prompt input_text such
        that the new prompt generates a response that is preferred as a
        response to input_text much less by a certain amount. This metric can
        be changed based on user needs.

        Args:
            input_text (str): input prompt to model that we want to explain
            epsilon_contrastive (float): amount of change in response to deem
                a contrastive explanation
            epsilon_iter (float): minimum amount of change between iterations
                to continue search
            split_k (int): number of words to be split into each token that
                is masked together
            info (boolean): True if to print output information, False
                otherwise
            ir (boolean): True if to do input reduction, i.e., remove tokens
                that cause minimal change to response until a large change
                occurs

        Returns:
            result (dico): contains various pieces of contrastive explanation
                including contrastive prompt, response to the contrastive
                prompt, response to the input prompt, and which words were
                modified
        """

        if info:
            if ir:
                print('Starting Input Reduction')
            else:
                print('Starting (myopic) Contrastive Explanations for Large Language Models')

        # NOTE: no initial classifiers are implemented but here is where one would put it
        if self._scalarizer_type == 'classifier':
            if self._scalarizer_name == 'hate':
                print('INVALID SCALARIZER FOR CLASSIFICATION TASK')
#                (scores_input_text, label_input_text) = self._scalarizer_func.scalarizer(input_text,  input_text, input_text, input_text, -1)
            else:
                print('INVALID SCALARIZER FOR CLASSIFICATION TASK')
        else:
            scores_input_text = 0
            label_input_text = -1

        output_text = self._model.generate(input_text, text_only=True, **model_params)[0] # output from input text prompt

        input_tokens = self.splitTextByK(input_text, split_k)
        num_input_tokens = len(input_tokens)

        tokens_changed = np.zeros((num_input_tokens,1)) # keep track of which tokens have been modified
        modify_token = True
        input_tokens_curr = input_tokens.copy()
        iters = 0
        scores_max_prev = 0
        count_no_change = 0
        mask_order = [] # keep track of order of tokens being masked
        masks_optimal = [] # keep track of the tokens that masked
        modifications_optimal = [] # keep track of the modifications made
        num_model_calls = 0
        while modify_token:
            print('Running iteration '+str(iters+1))
            inds_modify = np.where(tokens_changed == 0)[0] # tokens that have not yet been modified

            num_input_modify = len(inds_modify)
            scores = np.zeros((num_input_modify,1))
            scores_abs = np.zeros((num_input_modify,1))
            labels_contrast = np.zeros((num_input_modify,)) # for classification tasks
            prompts_modified = {}
            responses_modified = {}
            prompts_masked_enc = {}
            prompts_modified_enc = {}
            mask_filled_dico = {}
            for i in range(num_input_modify):
                input_tokens_mask = input_tokens_curr.copy()
                input_tokens_mask[inds_modify[i]] = self._infiller.mask_string
                input_text_mask = ' '.join(input_tokens_mask)

                batch = self._infiller.encode(input_text_mask, add_special_tokens=True)
                (generated_ids, mask_filled) = self._infiller.generate(batch, masked_word=input_tokens_curr[inds_modify[i]], num_return_sequences=self._num_return_sequences, return_mask_filled=True)
                input_text_infilled = self._infiller.decode(generated_ids)

                # these encodings are used later to find what was infilled for mask
                prompts_masked_enc[i] = batch
                prompts_modified_enc[i] = generated_ids
                mask_filled_dico[i] = mask_filled

                prompts_modified[i] = input_text_infilled
                output_infilled_text = self._model.generate(input_text_infilled, text_only=True, **model_params)[0] # output from modified input text prompt
                num_model_calls += 1
                responses_modified[i] = output_infilled_text

                (score_temp, label_temp) = self._scalarizer_func.scalarize_output(input_text,  output_text, input_text_infilled, output_infilled_text, input_label=label_input_text)
                scores[i] = score_temp
                labels_contrast[i] = label_temp
                if self._scalarizer_type == 'distance':
                    scores_abs[i] = np.abs(scores[i]) # measure the absolute difference
                else: # scalarizer_type is classifier so always want to measure in one direction
                    scores[i] = scores_input_text - scores[i] # classification always measures difference from input text score
                    scores_abs[i] = scores[i]

            if ir:
                inds_max = np.argmin(scores_abs)
            else:
                inds_max = np.argmax(scores_abs)

            scores_max = scores_abs[inds_max]
            tokens_changed[inds_modify[inds_max]] = 1
            mask_order.append(inds_modify[inds_max])
            # find what replaced the <mask>

            mask_filled = mask_filled_dico[inds_max]

            token_to_modify = input_tokens_curr[inds_modify[inds_max]]
            modifications_optimal.append(input_tokens_curr[inds_modify[inds_max]]+'->'+mask_filled)
            input_tokens_curr[inds_modify[inds_max]] = mask_filled
            masks_optimal.append(mask_filled)

            if ir:
                if scores_max > epsilon_contrastive and iters < (num_input_tokens-1):
                    modify_token = False
                    # remove previous modifications
                    input_tokens_curr[inds_modify[inds_max]] = token_to_modify
                    mask_order = mask_order[:-1]
                    masks_optimal = masks_optimal[:-1]
                    modifications_optimal = modifications_optimal[:-1]
            elif self._scalarizer_type == 'classifier':
                if np.abs(scores_max-scores_max_prev) <= epsilon_iter:
                    count_no_change += 1
                else:
                    count_no_change = 0
                if labels_contrast[inds_max] != label_input_text or count_no_change >= no_change_max_iters:
                    modify_token = False
                    if info:
                        if labels_contrast[inds_max] != label_input_text:
                            print('Stopping because initial classification has changed')
                        elif count_no_change >= no_change_max_iters:
                            print('Stopping because no significant change has occurred in '+str(no_change_max_iters)+ ' iterations.')
            else: # scalarizer_type is distance
                if np.abs(scores_max-scores_max_prev) <= epsilon_iter:
                    count_no_change += 1
                else:
                    count_no_change = 0
                if scores_max > epsilon_contrastive or count_no_change >= no_change_max_iters:
                    modify_token = False
                    if info:
                        if scores_max > epsilon_contrastive:
                            print('Stopping because contrastive threshold has been passed')
                        elif count_no_change >= no_change_max_iters:
                            print('Stopping because no significant change has occurred in '+str(no_change_max_iters)+ ' iterations.')

            if iters >= (num_input_tokens-1):
                modify_token = False
                if info:
                    print('Modified all tokens.')
            scores_max_prev = scores_max
            iters += 1

        prompt_contrastive = ' '.join(input_tokens_curr)

        if info:
            print(str(num_model_calls) + ' model calls made.')
            if ir:
                print('Input Reduction Solution')
            else:
                print('Contrastive Explanation Solution')
            print('Scalarizer: '+ self._scalarizer_name)
            print('Input prompt: ' + input_text)
            if self._generation:
                print('Input response: ' + output_text)
            print('Contrastive prompt: ' + prompt_contrastive)
            if self._generation:
                print('Contrastive response: ' + responses_modified[inds_max])
            print('Modifications made: ')
            for l in range(len(modifications_optimal)):
                print('        '+modifications_optimal[l])

            if self._scalarizer_name == 'preference':
                if scores[inds_max] > 0:
                    print('Preference decreased.')
                elif scores[inds_max] < 0:
                    print('Preference increased.')
                else:
                    print('Prefence remained the same.')
            elif self._scalarizer_name == 'nli' or self._scalarizer_name == 'contradiction':
                (score_temp, label_temp) = self._scalarizer_func.scalarize_output(input_text,  output_text, input_text, responses_modified[inds_max], input_label=label_input_text, info=True)
            elif self._scalarizer_name == 'bleu':
                print('BLEU score of difference in responses is larger than threshold.')
            elif self._scalarizer_name == 'implicit_hate' or self._scalarizer_name == 'stigma':
                print('Initial label: ' + self._scalarizer_func._model.config.id2label[label_input_text])
                print('Contrast label: ' + self._scalarizer_func._model.config.id2label[labels_contrast[inds_max]])
            else:
                    print('INVALID SCALARIZER')

        result = {}
        result['prompt_cell'] = prompt_contrastive
        result['response_cell'] = responses_modified[inds_max]
        result['output_original'] = output_text
        result['tokens_cell'] = input_tokens_curr
        result['mask_order'] = mask_order
        result['masks_optimal'] = masks_optimal

        return result
