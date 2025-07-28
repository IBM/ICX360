"""File containing class CELL

CELL is an explainer class that contains function explain_instance that
provides contrastive explanations of input instances. The algorithm for
providing explanations is described as CELL in: CELL your Model: Contrastive
Explanations for Large Language Models, Ronny Luss, Erik Miehling, Amit
Dhurandhar. https://arxiv.org/abs/2406.11785
"""

import random
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


class CELL(LocalBBExplainer):
    """Instances of CELL contain information about the LLM model being explained.
    These instances are used to explain LLM responses on input text using a
    budgeted algorithm with intelligent search strategy.

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
            num_return_sequences (int): number of sequences returned when
                doing generation for mask infilling
            scalarizer (str): select which scalarizer to use to determine
                if a contrast is found (must be from ['shp', 'nli', 'bleu'])
            scalarizer_model_path (str): allow user to pass a model path for
                scalarizers (e.g., choose 'stanfordnlp/SteamSHP-flan-t5-xl'
                instead of default 'stanfordnlp/SteamSHP-flan-t5-large')
            scalarizer_type (str): 'distance' for explaining LLM generation
                using distances, 'classifier' for explaining a classifier
            generation (bool): the model being explained performs true
                generation (as opposed to having output==input)
            experiment_id (str): passed to evaluate.load for certain
                scalarizers. This is used if several distributed
                evaluations share the same file system.
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
            raise Exception("CELL received parameter value for infiller that is not recognized")

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

    def explain_instance(self, input_text, epsilon_contrastive=.5, split_k=1, budget=100, radius=5, alpha=0.5, info=True, ir=False, input_text_list=[''], prompt_format = 'Context: $$input0$$ \n\nQuestion: $$input1$$ \n\nAnswer: ', multiple_inputs=False, input_inds_modify=[0], model_params={}):
        """Provide explanations of LLM applied to prompt input_text.

        Provide a contrastive explanation by changing prompt input_text such
        that the new prompt generates a response that is preferred as a
        response to input_text much less by a certain amount. This metric can
        be changed based on user needs.

        Args:
            input_text (str): input prompt to model that we want to explain
            epsilon_contrastive (float): amount of change in response to deem a
                contrastive explanation
            split_k (int): number of words to be split into each token that is
                masked together
            budget (int): maximum number of queries allowed from infilling model
            radius (int):  radius for sampling near a previously modified token
            alpha (float): tradeoff between exploration and exploitation. lower
                alpha mean more exploration, higher alpha means more
                exploitation
            info (bool): True if to print output information, False otherwise
            ir (bool): True if to do input reduction, i.e., remove tokens that
                cause minimal change to response until a large change occurs
            input_text_list (str list): if multiple_inputs==True, then use
                input_text_list to feed additional text segments
            prompt_format (str): format for prompt to create from input_text
                and input_text_list. Default is question/answering for
                google/flan-t5-large
            multiple_inputs (bool): True if example requires multiple inputs
                and a format, i.e., uses input_text and input_text_list, False
                if just input_text for prompt
            input_inds_modify (int list): list of which input_text segments to
                modify for contrastive example when multiple_inputs==True
            model_params (dico): additional keyword arguments for model
                generation (self._model.generate())

        Returns:
            result (dico): contains various pieces of contrastive explanation
                including contrastive prompt, response to the contrastive
                prompt, response to the input prompt, and which words were
                modified
        """

        if info:
            if ir:
                print('Starting Input Reduction')
                raise Exception("CELL needs to be implemented for ir=True")
            else:
                print('Starting Contrastive Explanations for Large Language Models')

        output_text = self._model.generate(input_text, text_only=True, **model_params)[0] # output from input text prompt

        # NOTE: no initial classifiers are implemented but here is where one would put it
        if self._scalarizer_type == 'classifier':
            if self._scalarizer_name == 'hate':
                print('INVALID SCALARIZER FOR CLASSIFICATION TASK')
 #                (scores_input_text, label_input_text) = self._scalarizer_func.scalarize_output(output_text,  output_text, output_text, output_text, -1)
            else:
                print('INVALID SCALARIZER FOR CLASSIFICATION TASK')
        else:
            scores_input_text = 0
            label_input_text = -1

        input_text_len = np.zeros((len(input_text_list)+1),)
        if multiple_inputs: # if there are multiple inputs, create the prompt appropriately
            input_text_len[0] = len(input_text.split(' '))
            input_text = prompt_format.replace('$$input0$$', input_text)
            for i in range(len(input_text_list)):
                input_text_len[i+1] = len(input_text_list[i].split(' '))
                input_text = input_text.replace('$$input'+str(i+1)+'$$', input_text_list[i])
            prompt_format_split = prompt_format.split(' ')

        if not multiple_inputs: # ToDo: Implement splitTextByK for multiple inputs
            input_tokens = self.splitTextByK(input_text, split_k)
        else:
            input_tokens = input_text.split(' ')
        num_input_tokens = len(input_tokens)

        tokens_changed = np.zeros((num_input_tokens,)) # keep track of which tokens have been modified
        if multiple_inputs: # adjust tokens_changed
            tokens_changed = -1*np.ones((num_input_tokens,)) # keep track of which tokens have been modified (-1 represents tokens to never change or focus on for sampling)
            for i in range(len(input_inds_modify)): # allow selected inputs to be modified for contrastive exmaple
                ind_modify = input_inds_modify[i]
                ind = prompt_format_split.index('$$input'+str(ind_modify)+'$$')
                for j in range(ind_modify):
                    ind += (input_text_len[j]-1) # subtract 1 for token $$inputX$$
                tokens_changed[int(ind):int(ind+input_text_len[ind_modify])] = 0

        modify_token = True
        input_tokens_curr = input_tokens.copy()
        iters = 0
        mask_order = [] # keep track of order of tokens being masked
        masks_optimal = [] # keep track of the tokens that masked
        modifications_optimal = [] # keep track of the modifications made
        q = int(np.floor(budget/np.log2(budget))) # parameter to keep track of query budget
        num_iters = int(np.floor(np.log2(budget)))+1 # maximum number of outer iterations
        k = 0 # parameter to determine sampling size
        # these lists keep track of different required structures, initialized for no tokens infilled
        prototypes_list_full = []
        scores_list_full = []
        label_list_full = []
        inds_not_sampled_arr = 1 - tokens_changed
        num_model_calls = 0
        budget_used = False
#        for i in range(num_iters):
        i = 0
        while modify_token and not budget_used:
            if modify_token == False: # a contrastive example was found
                break
            print('Running outer iteration '+str(i+1))
            if (i+1)*np.power(2,i+1) <= q:
                n = np.power(2,i+1)
                k = i+1
            else:
                n = np.power(2,k)
            m = n # we will sample m prototypes around which we will generate new potential prototypes
            # sample at least half from new positions if there are some left
            inds_not_sampled = list(np.where(inds_not_sampled_arr==1)[0])
            m_new_from_list = np.minimum(int(m*alpha), len(prototypes_list_full)) # sample from previously perturbed
            m_new_from_scratch = np.minimum(m - m_new_from_list, len(inds_not_sampled)) # sample from scratch

            inds_cont = random.sample(list(range(len(prototypes_list_full))), m_new_from_list) # indices of samples from previous list
            inds_scratch = random.sample(inds_not_sampled, m_new_from_scratch) # indices of new tokens to perturb

            prototypes_centers = []
            for ind in inds_scratch: # sample once from each ind
                if num_model_calls >= budget:
                    budget_used = True
                    break
                inds_not_sampled_arr[ind] = 0
                sample_scratch = {}
                sample_scratch['mask_order'] = []
                sample_scratch['masks_optimal'] = []
                sample_scratch['modifications_optimal'] = []
                sample_scratch['input_tokens'] = input_tokens.copy()
                sample_scratch['tokens_changed'] = tokens_changed.copy() # keep track of which tokens have been modified
                sample_scratch['scores'] = -999
                samples_temp = self.sample(sample_scratch, ind, 0, 1, model_params=model_params)
                prototypes_centers.extend(samples_temp)
                num_model_calls += len(samples_temp)
            for ind in inds_cont: # sample prototype centers
                if num_model_calls >= budget:
                    budget_used = True
                    break
                inds_focus_temp = list(np.where(prototypes_list_full[ind]['tokens_changed']==1)[0])
                ind_focus = random.sample(inds_focus_temp, 1)[0] # sample a token that has already been modified to then sample near that
                samples_temp = self.sample(prototypes_list_full[ind], ind_focus, radius, 1, model_params=model_params)
                prototypes_centers.extend(samples_temp)
                num_model_calls += len(samples_temp)

            prototypes_list_full.extend(prototypes_centers) # add new samples
            # pass all initial centers through scalarizer
            for j in range(len(prototypes_centers)):
                (score_temp, label_temp) = self._scalarizer_func.scalarize_output(input_text,  output_text, \
                    prototypes_centers[j]['prompts_modified'], prototypes_centers[j]['responses_modified'], input_label=label_input_text)
                scores_list_full.append(score_temp)
                label_list_full.append(label_temp)
            for j in range(int(np.ceil(np.log2(n)))):
                if num_model_calls >= budget:
                    budget_used = True
                    break
                if modify_token == False: # a contrastive example was found
                    break
                prototypes = []
                num_sample_inner = int(np.floor(q/m/np.ceil(np.log2(n))))
                for l in range(len(prototypes_centers)): # sample num_sample_inner per prototype center
                    inds_focus_temp = list(np.where(prototypes_centers[l]['tokens_changed']==1)[0])
                    ind_focus = random.sample(inds_focus_temp, 1)[0] # sample a token that has already been modified to then sample near that
                    samples_temp = self.sample(prototypes_centers[l], ind_focus, radius, num_sample_inner, model_params=model_params)
                    prototypes.extend(samples_temp)
                    num_model_calls += len(samples_temp)
                    if num_model_calls >= budget:
                        budget_used = True
                        break

                prototypes_list_full.extend(prototypes) # add new samples before adding centers to prototypes
                num_prototypes = len(prototypes)
                prototypes.extend(prototypes_centers) # check for scores when including prototype centers
                scores = np.zeros((len(prototypes),))
                scores_abs = np.zeros((len(prototypes),))
                labels_contrast = np.zeros((len(prototypes),)) # for classification tasks
                for l in range(len(prototypes)):
                    (score_temp, label_temp) = self._scalarizer_func.scalarize_output(input_text,  output_text, \
                        prototypes[l]['prompts_modified'], prototypes[l]['responses_modified'], input_label=label_input_text)
                    if l < num_prototypes: # append prototypes (not the centers) to list of scores/labels
                        scores_list_full.append(score_temp)
                        label_list_full.append(label_temp)
                    scores[l] = score_temp
                    labels_contrast[l] = label_temp
                    if self._scalarizer_type == 'distance':
                        scores_abs[l] = np.abs(scores[l]) # measure the absolute difference
                    else: # scalarizer_type is classifier so always want to measure in one direction
                        scores[l] = scores_input_text - scores[l] # classification always measures difference from input text score
                        scores_abs[l] = scores[l]
                if len(prototypes) == 0:
                    break;
                scores_max = np.max(scores_abs)
                ind_max = np.argmax(scores_abs)
                if ir:
                    raise Exception("CELL needs to be implemented for ir=True")
                    # if scores_max > epsilon_contrastive:
                    #     modify_token = False
                    #     output_sample = prototypes[ind_max]
                    #     output_score = scores[ind_max]
                    #     # remove previous modifications
                    #     input_tokens_curr[inds_modify[inds_max]] = token_to_modify
                    #     mask_order = mask_order[:-1]
                    #     masks_optimal = masks_optimal[:-1]
                    #     modifications_optimal = modifications_optimal[:-1]
                else:
                    if scores_max > epsilon_contrastive:
                        modify_token = False
                        output_sample = prototypes[ind_max]
                        output_score = scores[ind_max]
                        if info:
                            print('Stopping because contrastive threshold has been passed')
                if modify_token: # no contrastive example found so sample new prototype centers
                    inds_sorted = np.flip(np.argsort(scores_abs))
                    inds_sorted = inds_sorted[0:int(np.ceil(m/2))] # select indices with highest contrastive scores
                    prototypes_centers = [prototypes[l] for l in list(inds_sorted)]
                    m = int(np.ceil(m/2))
            if len(prototypes) == 0:
                print('CELL WARNING: No more prototypes to search.')
                break;
            i += 1 # outer iteration counter

        if len(prototypes) == 0 and modify_token and not budget_used:
            print('No solution found.') # there is still a budget
        elif modify_token and budget_used:
            print('Used up budget and no solution found.')
        elif modify_token:
            print('No solution found.')
        elif budget_used:
            print('Used up budget.')
        if info:
            print(str(num_model_calls) + ' model calls made.')
            if modify_token:
                print('Results of best example found follow:')
                scores = np.array(scores_list_full)
                if self._scalarizer_type == 'distance':
                    scores_abs = np.abs(scores) # measure the absolute difference
                else: # task is classification so always want to measure in one direction
                    scores = scores_input_text - scores # classification always measures difference from input text score
                    scores_abs = scores
                ind_max = np.argmax(scores_abs)
                output_sample = prototypes_list_full[ind_max]
            if ir:
                print('Input Reduction Solution')
            else:
                print('Contrastive Explanation Solution')
            print('Scalarizer: '+ self._scalarizer_name)
            print('Input prompt: ' + input_text)
            if self._generation:
                print('Input response: ' + output_text)
            print('Contrastive prompt: ' + output_sample['prompts_modified'])
            if self._generation:
                print('Contrastive response: ' + output_sample['responses_modified'])
            print('Modifications made: ')
            for l in range(len(output_sample['modifications_optimal'])):
                print('        '+output_sample['modifications_optimal'][l])
            if self._scalarizer_name == 'preference':
                if scores[ind_max] > 0:
                    print('Preference decreased.')
                elif scores[ind_max] < 0:
                    print('Preference increased.')
                else:
                    print('Prefence remained the same.')
            elif self._scalarizer_name == 'nli' or self._scalarizer_name == 'contradiction':
                (score_temp, label_temp) = self._scalarizer_func.scalarize_output(input_text,  output_text, input_text, output_sample['responses_modified'], input_label=label_input_text, info=True)   # run nli model with two outputs
            elif self._scalarizer_name == 'bleu':
                if modify_token == False:
                    print('BLEU score of difference in responses is larger than threshold.')
                else:
                    print('BLEU score of difference in responses is not larger than threshold.')
            else:
                    print('INVALID SCALARIZER')

        result = {}
        result['prompt_cell'] = output_sample['prompts_modified']
        result['response_cell'] = output_sample['responses_modified']
        result['output_original'] = output_text
        result['tokens_cell'] = output_sample['input_tokens']
        result['mask_order'] = output_sample['mask_order']

        return result

    def sample(self, input_sample, curr_position, radius, num_samples, model_params={}):
        """Generate sample prompts based on an input prompt

        Args:
            input_sample (dico): contains information about a prompt including
                text and how it differs from the input prompt to the explainer
            curr_position (int): position of tokens from which to generate
                samples within a radius of
            radius (int):  radius for sampling near a previously modified token
            num_samples (int): number of samples to generate
            model_params (dico): additional keyword arguments for model
                generation (self._model.generate())

        Returns:
            samples_list (dico list): list of samples which are dictionaries
                with same information as input_sample
        """

        inds_modify = np.where(input_sample['tokens_changed'] == 0)[0] # tokens that have not yet been modified
        inds_modify_restricted = inds_modify[np.where(np.abs(inds_modify-curr_position) <= radius)] # sample only from inds within radius of curr_position
        inds_modify_selected = random.sample(list(inds_modify_restricted), np.minimum(num_samples, len(list(inds_modify_restricted)))) # sample num_samples words around curr_position
        samples = {}
        for i in range(len(inds_modify_selected)):
            samples[i] = {}

            input_tokens_curr = input_sample['input_tokens'].copy()
            samples[i]['tokens_changed'] = input_sample['tokens_changed'].copy()
            samples[i]['tokens_changed'][inds_modify_selected[i]] = 1
            samples[i]['mask_order'] = input_sample['mask_order'].copy() # keep track of order of tokens being masked
            samples[i]['mask_order'].append(inds_modify_selected[i])

            input_tokens_mask = input_tokens_curr.copy()
            input_tokens_mask[inds_modify_selected[i]] = self._infiller.mask_string
            input_text_mask = ' '.join(input_tokens_mask)

            batch = self._infiller.encode(input_text_mask, add_special_tokens=True)
            (generated_ids, mask_filled) = self._infiller.generate(batch, masked_word=input_tokens_curr[inds_modify_selected[i]], num_return_sequences=self._num_return_sequences, return_mask_filled=True)
            input_text_infilled = self._infiller.decode(generated_ids)

            # these encodings are used later to find what was infilled for mask
            samples[i]['mask_filled'] = mask_filled

            samples[i]['prompts_modified'] = input_text_infilled
            output_infilled_text = self._model.generate(input_text_infilled, text_only=True, **model_params)[0] # output from modified input text prompt
            samples[i]['responses_modified'] = output_infilled_text

            samples[i]['masks_optimal'] = input_sample['masks_optimal'].copy() # keep track of the tokens that masked
            samples[i]['modifications_optimal'] = input_sample['modifications_optimal'].copy() # keep track of the modifications made
            # find what replaced the <mask>
            token_to_modify = input_tokens_curr[inds_modify_selected[i]]
            samples[i]['modifications_optimal'].append(input_tokens_curr[inds_modify_selected[i]]+'->'+mask_filled)
            input_tokens_curr[inds_modify_selected[i]] = mask_filled
            samples[i]['input_tokens'] = input_tokens_curr.copy()
            samples[i]['masks_optimal'].append(mask_filled)

        samples_list = [samples[i] for i in range(len(inds_modify_selected))]
        return samples_list
