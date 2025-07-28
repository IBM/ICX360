"""
Scalarized model that computes the log probability of generating a reference output conditioned on inputs.

This "scalarized model" is a generative model that can also compute the log probability (or a transformation thereof)
of generating a given reference output conditioned on inputs.
"""
# Assisted by watsonx Code Assistant in formatting and augmenting docstrings.

from math import ceil, log2

import torch

from icx360.utils.model_wrappers import HFModel, VLLMModel
from icx360.utils.scalarizers import Scalarizer
from icx360.utils.toma import toma_get_probs


class ProbScalarizedModel(Scalarizer):
    """
    Generative model that also computes the probability of a given reference output conditioned on inputs.

    Attributes:
        model (icx360.utils.model_wrappers.Model):
            Generative model, wrapped in an icx360.utils.model_wrappers.Model object.
    """
    def __init__(self, model):
        """
        Initialize ProbScalarizedModel.

        Args:
            model (icx360.utils.model_wrappers.Model):
                Generative model, wrapped in an icx360.utils.model_wrappers.Model object.

        Raises:
            TypeError: If the model is not an icx360.utils.model_wrappers.HFModel
                or an icx360.utils.model_wrappers.VLLMModel.
        """
        super().__init__(model)
        if not isinstance(model, HFModel) and not isinstance(model, VLLMModel):
            raise TypeError("Model must be a HFModel (HuggingFace) or VLLMModel for ProbScalarizedModel")

    def scalarize_output(self, inputs=None, outputs=None, ref_input=None, ref_output=None, chat_template=False, system_prompt=None, tokenizer_kwargs={}, transformation="log_prob_mean", **kwargs):
        """
        Compute probability of reference output conditioned on inputs.

        Args:
            inputs (str or List[str] or List[List[str]]):
                Inputs to compute probabilities for:
                A single input text, a list of input texts, or a list of segmented texts.
            outputs (str or List[str] or None):
                Outputs to scalarize (corresponding to inputs) - not used.
            ref_input (str or None):
                Reference input used to scalarize - not used.
            ref_output (icx360.utils.model_wrappers.GeneratedOutput):
                Reference output object.
            chat_template (bool):
                Whether to apply chat template.
            system_prompt (str or None):
                System prompt to include in chat template.
            tokenizer_kwargs (dict):
                Additional keyword arguments for tokenizer.
            transformation (str, optional):
                Transformation to apply to token probabilities.
                    "log_prob_mean": arithmetic mean of log probabilities (default).
                    "log_prob_sum": sum of log probabilities.
                    "prob_geo_mean": geometric mean of probabilities.
                    "prob_prod": product of probabilities.
            **kwargs (dict):
                Additional keyword arguments for model.

        Returns:
            probs_transformed ((num_inputs,) torch.Tensor):
                Transformed probability of generating the reference output conditioned on each input.
        """
        # Check for and convert inputs
        if inputs is None:
            raise ValueError("inputs must be provided for ProbScalarizedModel.scalarize_output()")
        else:
            inputs = self.model.convert_input(inputs, chat_template, system_prompt, **tokenizer_kwargs)
        # Check for reference output
        if ref_output is None:
            raise ValueError("ref_output must be provided for ProbScalarizedModel.scalarize_output()")

        # Compute log probabilities of reference output tokens conditioned on inputs
        if isinstance(self.model, HFModel):
            log_probs = self._compute_log_probs_hf(inputs, ref_output, **kwargs)
        elif isinstance(self.model, VLLMModel):
            log_probs = self._compute_log_probs_vllm(inputs, ref_output, **kwargs)

        # Transform probabilities
        if transformation in ("log_prob_mean", "prob_geo_mean"):
            # Mean of log probabilities
            probs_transformed = log_probs.mean(dim=1)
        elif transformation in ("log_prob_sum", "prob_prod"):
            # Sum of log probabilities
            probs_transformed = log_probs.sum(dim=1)
        else:
            raise ValueError("Transformation not recognized")
        if transformation.startswith("prob"):
            # Convert log probabilities to probabilities
            probs_transformed = probs_transformed.exp()

        return probs_transformed

    def _compute_log_probs_hf(self, inputs, ref_output, **kwargs):
        """
        Compute log probabilities of reference output tokens conditioned on inputs for an HFModel.

        Args:
            inputs (transformers.BatchEncoding):
                BatchEncoding of inputs produced by tokenizer.
            ref_output (icx360.utils.model_wrappers.GeneratedOutput):
                Reference output object containing a sequence of token IDs (ref_output.output_ids).
            **kwargs (dict):
                Additional keyword arguments for model.

        Returns:
            log_probs ((num_inputs, gen_length) torch.Tensor):
                Log probabilities of reference output tokens.
        """
        num_inputs = inputs["input_ids"].shape[0]
        # Get token IDs of reference output
        ref_output = ref_output.output_ids

        # Number of generated tokens in output
        # encoder-decoder output always begins with a fixed special token e.g. <pad>,
        # while decoder-only output has been truncated to only the generated response
        gen_length = ref_output.shape[1] - self.model._model.config.is_encoder_decoder

        if num_inputs == 1 or not torch.cuda.is_available():
            # Call underlying HuggingFace model on given input and output sequences to obtain logits
            ref_output_expanded = ref_output.expand(num_inputs, -1)
            with torch.no_grad():
                if self.model._model.config.is_encoder_decoder:
                    # Encoder-decoder model: pass inputs and reference output as separate arguments
                    output_dict = self.model._model(**inputs, decoder_input_ids=ref_output_expanded)
                else:
                    # Decoder-only model: concatenate inputs with reference output
                    combined_input_output = torch.cat([inputs["input_ids"], ref_output_expanded], dim=1)
                    output_dict = self.model._model(combined_input_output)

            # Position where generated output starts (in concatenated input-output for decoder-only)
            gen_start = 1 if self.model._model.config.is_encoder_decoder else inputs["input_ids"].shape[1]
            # Convert logits into tuple
            # logits indices are off by one because logits at position i-1 are for predicting token at position i
            scores = tuple(output_dict.logits[:, pos, :] for pos in range(gen_start - 1, gen_start + gen_length - 1))

            # Compute probabilities of tokens in reference output
            # NOTE: although ref_output_expanded and scores have different token lengths,
            # compute_transition_scores() seems to align their last positions
            log_probs = self.model._model.compute_transition_scores(ref_output_expanded, scores, normalize_logits=True)

        else:
            # Call using toma
            # Pre-allocate log_probs Tensor
            log_probs = torch.empty((num_inputs, gen_length), device=self.model._device)

            # Call using toma
            batch_size_init = 2 ** ceil(log2(num_inputs))
            toma_get_probs(0, num_inputs, self.model._model, inputs, ref_output, log_probs, toma_initial_step=batch_size_init)

        return log_probs

    def _compute_log_probs_vllm(self, inputs, ref_output, **kwargs):
        """
        Compute log probabilities of reference output tokens conditioned on inputs for a VLLMModel.

        Args:
            inputs (List[str]):
                Inputs to compute probabilities for.
            ref_output (icx360.utils.model_wrappers.GeneratedOutput):
                Reference output object containing reference text (ref_output.output_text[0]).
            **kwargs (dict):
                Additional keyword arguments for model.

        Returns:
            log_probs ((num_inputs, gen_length) torch.Tensor):
                Log probabilities of reference output tokens.
        """
        # VLLM parameters for computing log probs of a given input + output without generating
        kwargs["logprobs"] = 0
        kwargs["max_tokens"] = 0
        kwargs["echo"] = True

        # Call underlying VLLM model on inputs only to get their token lengths
        completion = self.model._model.completions.create(model=self.model._model_name, prompt=inputs, **kwargs)
        input_lengths = []
        for result in completion.choices:
            input_lengths.append(len(result.logprobs.tokens))

        # Combined inputs + output
        combined_input_output = [inp + ref_output.output_text[0] for inp in inputs]

        # Call VLLM model on combined inputs + output to get log probs
        completion = self.model._model.completions.create(model=self.model._model_name, prompt=combined_input_output, **kwargs)
        log_probs = []
        for i, result in enumerate(completion.choices):
            log_probs.append(result.logprobs.token_logprobs[input_lengths[i]:])

        return torch.tensor(log_probs)
