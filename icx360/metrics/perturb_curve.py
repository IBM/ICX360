"""
Perturbation curve evaluator for measuring the fidelity of input attributions to the explained model.

The PerturbCurveEvaluator class evaluates perturbation curves for input attribution scores
produced by icx360.algorithms.mexgen.CLIME.explain_instance() or icx360.algorithms.mexgen.LSHAP.explain_instance().
It thus evaluates the fidelity of these attribution scores to the explained model.
"""
# Assisted by watsonx Code Assistant in formatting and augmenting docstrings.

from math import ceil

import numpy as np
import torch

from icx360.utils.scalarizers import ProbScalarizedModel, TextScalarizedModel
from icx360.utils.subset_utils import mask_subsets


class PerturbCurveEvaluator:
    """
    Perturbation curve evaluator for measuring the fidelity of input attributions to the explained model.

    Attributes:
        model (icx360.utils.model_wrappers.Model):
            Model to explain, wrapped in an icx360.utils.model_wrappers.Model object.
        scalarized_model (icx360.utils.scalarizers.Scalarizer):
            "Scalarized model" that further wraps `model` with a method for computing scalar values
            based on the model's inputs or outputs.
    """
    def __init__(self, model, scalarizer="prob", **kwargs):
        """
        Initialize perturbation curve evaluator.

        Args:
            model (icx360.utils.model_wrappers.Model):
                Model to explain, wrapped in an icx360.utils.model_wrappers.Model object.
            scalarizer (str):
                Type of scalarizer to use.
                    "prob": probability of generating original output conditioned on perturbed inputs
                        (instantiates an icx360.utils.scalarizers.ProbScalarizedModel).
                    "text": similarity scores between original output and perturbed outputs
                        (instantiates an icx360.utils.scalarizers.TextScalarizedModel).
            **kwargs (dict):
                Additional keyword arguments for initializing scalarizer.

        Raises:
            ValueError: If `scalarizer` is not "prob" or "text".
        """
        self.model = model

        # Instantiate scalarized model
        if scalarizer == "prob":
            self.scalarized_model = ProbScalarizedModel(model)
        elif scalarizer == "text":
            self.scalarized_model = TextScalarizedModel(model, **kwargs)
        else:
            raise ValueError("Scalarizer not supported")

    def eval_perturb_curve(self, explainer_dict, score_label, token_frac=False, max_frac_perturb=0.5,
                           replacement_str="", model_params={}, scalarize_params={}):
        """
        Evaluate perturbation curve for given input attributions.

        This method evaluates the perturbation curve for a set of attribution scores
        by perturbing units in decreasing order of their attribution scores.

        Args:
            explainer_dict (dict):
                Attribution dictionary as produced by icx360.algorithms.mexgen.CLIME.explain_instance()
                or icx360.algorithms.mexgen.LSHAP.explain_instance().
            score_label (str):
                Label of the attribution score to use for ranking units.
            token_frac (bool, optional):
                Whether to consider the number of tokens in each unit when ranking and perturbing units.
                Defaults to False.
            max_frac_perturb (float, optional):
                Maximum fraction of units or tokens to perturb. Defaults to 0.5.
            replacement_str (str, optional):
                String to replace perturbed units with. Defaults to "" for dropping units.
            model_params (dict, optional):
                Additional keyword arguments for model generation (for the self.model.generate() method).
            scalarize_params (dict, optional):
                Additional keyword arguments for computing scalar outputs
                (for the self.scalarized_model.scalarize_output() method).

        Returns:
            output_perturbed (dict):
                Dictionary with the following items:
                    "frac" (torch.Tensor):
                        Fractions of units or tokens perturbed.
                    `score_label` (torch.Tensor):
                        One or more Tensors of scalarized output values corresponding to the fractions
                        in the "frac" Tensor. `score_label` labels each Tensor with the type of scalarizer.

        Raises:
            ValueError:
                If token_frac is True and model's tokenizer is not available.
        """
        # 0) Extract original input units, unit types, attribution scores
        input_orig = explainer_dict["attributions"]["units"]
        unit_types = np.array(explainer_dict["attributions"]["unit_types"])
        scores = explainer_dict["attributions"][score_label]
        num_units = len(input_orig)
        # Avoid choosing units of type "n" for perturbation
        scores[unit_types == "n"] = -np.inf

        if token_frac:
            if self.model._tokenizer is None:
                raise ValueError("If token_frac==True, need to include model's tokenizer in model wrapper to count numbers of tokens")

            # 1) Count number of tokens in each unit
            # First encode segmented input
            input_encoding = self.model._tokenizer(input_orig, is_split_into_words=True, return_tensors="pt")
            # Iterate over units
            num_tokens = np.zeros(num_units, dtype=int)
            for u in range(num_units):
                # Token span and token count of unit
                token_span = input_encoding.word_to_tokens(u)
                if token_span is not None:
                    num_tokens[u] = token_span.end - token_span.start
            # Total number of tokens in units that can be perturbed
            tot_tokens_can_perturb = num_tokens[unit_types != "n"].sum()

            # 2) Sort units in increasing order of attribution scores divided by token counts
            scores_argsort = np.argsort(scores / num_tokens)

            # 3) Determine subsets of units to perturb
            # Initialize fractions to perturb and subsets to perturb
            k = 0
            frac_perturbed = [0.0]
            subsets_replace = [[]]
            # Increase fractions and subsets until fraction exceeds maximum specified
            while frac_perturbed[-1] < max_frac_perturb:
                k += 1
                frac_perturbed.append(num_tokens[scores_argsort[-k:]].sum() / tot_tokens_can_perturb)
                subsets_replace.append(np.sort(scores_argsort[-k:]))

        else:
            # 1) Number of units that can be perturbed
            num_units_can_perturb = (unit_types != "n").sum()

            # 2) Sort units in increasing order of attribution scores
            scores_argsort = np.argsort(scores)

            # 3) Determine subsets of units to perturb
            # Maximum number of units to perturb
            max_num_perturb = ceil(max_frac_perturb * num_units_can_perturb)
            # Fractions to perturb
            frac_perturbed = torch.arange(max_num_perturb + 1) / num_units_can_perturb
            # Subsets to perturb
            subsets_replace = [np.sort(scores_argsort[num_units - k:]) for k in range(max_num_perturb + 1)]

        # 4) Replace subsets of units with replacement string
        input_perturbed = mask_subsets(input_orig, subsets_replace, replacement_str)

        # 5) Compute model's scalarized outputs for perturbed inputs
        output_perturbed = self.scalarized_model.scalarize_output(inputs=input_perturbed, ref_output=explainer_dict["output_orig"], **model_params, **scalarize_params)

        # Move scalarized outputs to CPU
        if type(output_perturbed) is dict:
            for key in output_perturbed:
                output_perturbed[key] = output_perturbed[key].cpu()
        else:
            # Convert to dictionary
            if isinstance(self.scalarized_model, ProbScalarizedModel):
                output_perturbed = {"prob": output_perturbed.cpu()}
            else:
                output_perturbed = {"score": output_perturbed.cpu()}
        # Add fractions to dictionary
        output_perturbed["frac"] = torch.tensor(frac_perturbed)

        return output_perturbed
