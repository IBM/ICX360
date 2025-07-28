"""File containing class BART_infiller

BART_infiller is used to perform infilling using a BART LLM.
"""

import re

import numpy as np
import torch
from nltk.stem.porter import PorterStemmer
from transformers import BartForConditionalGeneration, BartTokenizer


class BART_infiller():
    """BART_infiller object.

    Instances can be used to encode, decode, and generate text to infill
    masks in text.

    Attributes
        _model: BART model used for infilling
        _tokenizer: BART tokenizer
        mask_string: text that represents a mask for BART
        mask_string_encoded: encoded version of mask for BART
        mask_filled_error: text representing that an infilling error occurred
    """

    def __init__(self, model_path="facebook/bart-large", device='cuda'):
        """Initialize BART infilling object.

        Args:
            model_path (str): name of BART model to be used for infilling
        """
        self._model = BartForConditionalGeneration.from_pretrained(model_path, device_map=device, forced_bos_token_id=0)
        self._tokenizer = BartTokenizer.from_pretrained(model_path, device_map=device)
        self._device = device
        self.mask_string = '<mask>'
        self.mask_string_encoded = self._tokenizer.encode(self.mask_string, add_special_tokens=False)[0]
        self.mask_filled_error = '!!abcxyz!!'

    def encode(self, text, add_special_tokens=False):
        """Function to encode text via BART tokenizer

        Args:
            text (str): string to encode
            add_special_tokens (bool): True if to use special tokens in
                encoding

        Returns:
            ret (int list): token indices where n is based on input text
        """

        ret = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return ret

    def decode(self, tokens, skip_special_tokens=True):
        """Function to decode text via BART tokenizer

        Args:
            tokens (int list): token indices
            skip_special_tokens (bool): True if to skip special tokens in
                decoding

        Returns:
            ret (str): string frame decoding all input tokens
        """

        ret = self._tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
        return ret

    def generate(self, tokens, num_return_sequences=1, masked_word = '', return_mask_filled=False):
        """Generate text to infill mask tokens. Assumes one of tokens is
            <mask> which is token id self.mask_string_encoded
        Args:
            tokens (int list): token indices
            num_return_sequences (int): number of generations to return
                (default: 1)
            masked_word (str): word that is masked in tokens (default: '')
            return_mask_filled (bool): if true, return (ret, mask_filled),
                else return only ret

        Returns:
            ret (int list): list of token indices after calling model.generate
                on input tokens
            mask_filled (str): decoded version of infilled texts
        """

        max_new_tokens = len(tokens) + 20
        ret_list = self._model.generate(torch.tensor([tokens]).to(self._device), max_new_tokens=max_new_tokens, num_return_sequences=num_return_sequences, num_beams=num_return_sequences).tolist()
        no_mask_found = True # indicator to follow if only finding the same as masked_word
        if num_return_sequences == 1:
            ret = ret_list[0]
            (mask_filled, inds_infill) = self.get_infilled_mask(tokens, ret, return_tokens=True)
            ind_mask = tokens.index(self.mask_string_encoded)
            ret = tokens[0:ind_mask] + inds_infill + tokens[(ind_mask+1):]
            no_mask_found = False
        else:
            for i in range(len(ret_list)):
                ret = ret_list[i]
                (mask_filled, inds_infill) = self.get_infilled_mask(tokens, ret, return_tokens=True)
                if not self.similar(masked_word, mask_filled) and mask_filled != self.mask_filled_error:
                    ind_mask = tokens.index(self.mask_string_encoded)
                    ret = tokens[0:ind_mask] + inds_infill + tokens[(ind_mask+1):]
                    no_mask_found = False
                    break
        if no_mask_found:
            # if the word is still the same, returns the same word
            ind_mask = tokens.index(self.mask_string_encoded)
            ret = tokens[0:ind_mask] + inds_infill + tokens[(ind_mask+1):]
            mask_filled = masked_word
        if return_mask_filled:
            return (ret, mask_filled)
        else:
            return ret


    def get_infilled_mask(self, x_enc, y_enc, return_tokens=False):
        """Retrieve text that replaced <mask> when infilling from generation
            output

        Args:
            x_enc (int list): token indices where one token is <mask>, i.e.
                input to generation function
            y_enc (int list): token indices representing same as x_enc with
                several tokens replacing <mask>, i.e., output of generation
                function
            return_tokens (bool): if true, return (mask_filled, inds_infill),
                else return only mask_filled

        Returns:
            mask_filled (str): decoded tokens that replace <mask> in y_enc
                from x_enc
            inds_infill (int list): token indices representing encoded version
                of infilled text
        """

        ind_mask = x_enc.index(self.mask_string_encoded)
        return_found = True
        inds = np.where(np.array(y_enc[0:(ind_mask+1)])==x_enc[ind_mask-1])[0]
        if len(inds) > 0:
            ind_infill_start = inds[-1] # accounts for word before mask occurring multiple times
        else:
            print("BART_infiller.get_infilled_mask could not find token before <mask>")
            return_found = False
        try:
            ind_infill_end = y_enc[ind_infill_start:].index(x_enc[ind_mask+1])+ind_infill_start # accounts for special tokens that occur at beginning and end
        except ValueError:
            print("BART_infiller.get_infilled_mask could not find token after <mask>")
            return_found = False
        if return_found:
            inds_infill = y_enc[(ind_infill_start+1):ind_infill_end]
            mask_filled = self._tokenizer.decode(inds_infill).strip() # strip() added because tokens can have whitespace, i.e., " should" and "should" are two different tokens
        else:
            inds_infill = [-1]
            mask_filled = self.mask_filled_error
        if return_tokens:
            return (mask_filled, inds_infill)
        else:
            return mask_filled

    def similar(self, word, fill_in):
        """Determine if word is similar to fill_in

        Args:
            word (str): words to search for
            fill_in (str): filled in text to search for word in

        Returns:
            ret (bool): True if word is similar to fill_in, False otherwise
        """

        ret = False

        # remove all punctuations of interest
        word = re.sub(r'[.,!?\']','', word)
        fill_in = re.sub(r'[.,!?\']','', fill_in)

        # make all lowercase
        word = word.lower()
        fill_in = fill_in.lower()

        # stem all words and check if word is in fill_in
        word_list = word.split(' ')
        word_stemmed_list = [PorterStemmer().stem(word_list[i]) for i in range(len(word_list))]
        fill_in_list = fill_in.split(' ')
        fill_in_list_stemmed = [PorterStemmer().stem(fill_in_list[i]) for i in range(len(fill_in_list))]

        count = 0
        for word_stemmed in word_stemmed_list:
            if word_stemmed in fill_in_list_stemmed:
                count += 1
        if float(count/len(word_stemmed_list)) >= 0.75: # if 3/4 of the words occur in mask
            ret = True

        return ret
