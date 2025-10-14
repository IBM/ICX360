"""
Class and supporting functions for MExGen L-SHAP explainer.

The MExGen framework and L-SHAP algorithm are described in:
    Multi-Level Explanations for Generative Language Models.
    Lucas Monteiro Paes and Dennis Wei et al.
    The 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025).
    https://arxiv.org/abs/2403.14459
"""
# Assisted by watsonx Code Assistant in formatting and augmenting docstrings.

from math import comb

import numpy as np

from icx360.algorithms.mexgen import MExGenExplainer
from icx360.utils.scalarizers import ProbScalarizedModel, TextScalarizedModel
from icx360.utils.subset_utils import mask_subsets, sample_subsets


class LSHAP(MExGenExplainer):
    """
    MExGen L-SHAP explainer

    Attributes:
        model (icx360.utils.model_wrappers.Model):
            Model to explain, wrapped in an icx360.utils.model_wrappers.Model object.
        segmenter (icx360.utils.segmenters.SpaCySegmenter):
            Object for segmenting input text into units using a spaCy model.
        scalarized_model (icx360.utils.scalarizers.Scalarizer):
            "Scalarized model" that further wraps `model` with a method for computing scalar values
            based on the model's inputs or outputs.
    """
    def explain_instance(self, input_orig, unit_types="p", ind_interest=None, output_orig=None,
                         ind_segment=True, segment_type="s", max_phrase_length=10, segment_type_output=None,
                         model_params={}, scalarize_params={},
                         num_neighbors=2, max_units_replace=2, replacement_str=""):
        """
        Explain model output by attributing it to parts of the input text.

        Uses an algorithm called L-SHAP (a variant of SHAP)
        that computes approximate Shapley values as attribution scores.

        Args:
            input_orig (str or List[str]):
                [input] Input text as a single unit (if str) or segmented sequence of units (List[str]).
            unit_types (str or List[str]):
                [input] Types of units in input_orig.
                    "p" for paragraph, "s" for sentence, "w" for word,
                    "n" for not to be perturbed/attributed to.
                If str, applies to all units in input_orig, otherwise unit-specific.
            ind_interest (bool or List[bool] or None):
                [input] Indicator of units to attribute to ("of interest").
                Default None means np.array(unit_types) != "n".
            output_orig (str or List[str] or icx360.utils.model_wrappers.GeneratedOutput or None):
                [output] Output for original input.
                Can be a single unit (str), segmented into units (List[str]), a GeneratedOutput object, or None.
            ind_segment (bool or List[bool]):
                [segmentation] Whether to segment input text.
                If bool, applies to all units; if List[bool], applies to each unit individually.
            segment_type (str):
                [segmentation] Type of units to segment into: "s" for sentences, "w" for words, "ph" for phrases.
            max_phrase_length (int):
                [segmentation] Maximum phrase length in terms of spaCy tokens (default 10).
            segment_type_output (str or None):
                [segmentation] Type of units to segment output text into:
                "s" for sentences, "ph" for phrases, None for no segmentation.
            model_params (dict):
                Additional keyword arguments for model generation (for the self.model.generate() method).
            scalarize_params (dict):
                Additional keyword arguments for computing scalar outputs
                (for the self.scalarized_model.scalarize_output() method).
            num_neighbors (int):
                [perturbation] Number of neighbors on either side of unit of interest that can be perturbed.
                Default 2 (as an example) means two neighbors to the left AND two neighbors to the right.
            max_units_replace (int):
                [perturbation] Maximum number of units to perturb at one time (default 2).
            replacement_str (str):
                [perturbation] String to replace units with (default "" for dropping units).

        Returns:
            output_dict (dict):
                Dictionary with the following items:
                    "attributions" (dict):
                        Dictionary with attribution scores, corresponding input units, and unit types.
                    "output_orig" (icx360.utils.model_wrappers.GeneratedOutput):
                        Output object generated from original input.

                Items in "attributions" dictionary:
                    "units" (List[str]):
                        input_orig segmented into units if not already, otherwise same as original.
                    "unit_types" (List[str]):
                        Types of units.
                    `score_label` ((num_units,) np.ndarray):
                        One or more sets of attribution scores (labelled by the type of scalarizer).
        """
        # 1) Segment input text if needed
        input_orig, unit_types = self.segment_input(input_orig, unit_types, ind_segment, segment_type, max_phrase_length)
        num_units = len(input_orig)

        if ind_interest is None:
            # Default is to attribute to all units that can be perturbed
            ind_interest = np.array(unit_types) != "n"
        elif type(ind_interest) is bool:
            ind_interest = [ind_interest] * num_units

        # Indices of units of interest and indices that can be perturbed/replaced
        idx_interest = ind_interest.nonzero()[0]
        idx_replace = (np.array(unit_types) != "n").nonzero()[0]

        # 2) Generate output for original input or wrap provided output
        output_orig = self.generate_or_wrap_output(input_orig, output_orig, model_params)
        # Segment output text if needed
        output_orig = self.segment_output(output_orig, segment_type_output, max_phrase_length)
        num_output_units = 1 if type(output_orig.output_text[0]) is str else len(output_orig.output_text[0])

        # 3) Initialize quantities
        # Initialize importance scores
        if isinstance(self.scalarized_model, TextScalarizedModel):
            # Dictionary of importance scores, each corresponding to a kind of similarity
            importance_scores = {}
            for key in self.scalarized_model.sim_scores:
                importance_scores[key] = np.zeros(num_units)
        else:
            importance_scores = np.zeros((num_units, num_output_units))

        # Initialize quantities associated with units of interest
        idx_replace_i = [None] * len(idx_interest)
        subsets_replace_excl_interest = [None] * len(idx_interest)
        subsets_replace_incl_interest = [None] * len(idx_interest)
        subsets_replace_all = set()

        # Iterate over units of interest
        for i in range(len(idx_interest)):

            # 4) Adapt set of units that can be replaced
            idx_replace_i[i] = adapt_replacement_set(idx_replace, idx_interest[i], num_neighbors)

            # 5) Enumerate subsets of units to replace
            # Subsets excluding unit of interest
            subsets_replace_excl_interest[i] = sample_subsets(idx_replace_i[i], max_units_replace)
            # Same subsets including unit of interest
            subsets_replace_incl_interest[i] = [np.union1d(subset, idx_interest[i]).tolist() for subset in subsets_replace_excl_interest[i]]
            # Add unit of interest as singleton
            subsets_replace_incl_interest[i].insert(0, [idx_interest[i]])

            # Add to subsets to replace for all units of interest (if not already present)
            subsets_replace_all = subsets_replace_all.union(map(tuple, subsets_replace_excl_interest[i]))
            subsets_replace_all = subsets_replace_all.union(map(tuple, subsets_replace_incl_interest[i]))

        # 6) Replace subsets of units with replacement string
        # First convert subsets_replace_all to list of lists
        subsets_replace_all = list(map(list, subsets_replace_all))
        input_perturbed = mask_subsets(input_orig, subsets_replace_all, replacement_str)

        # 7) Compute scalarized outputs for perturbed inputs
        # Prepend original input
        input_perturbed.insert(0, input_orig)
        subsets_replace_all.insert(0, [])

        if "ref_output" not in scalarize_params:
            # Reference output is original output if not specified
            scalarize_params = scalarize_params.copy()
            scalarize_params["ref_output"] = output_orig

        # Compute scalarized outputs
        scalar_outputs = self.scalarized_model.scalarize_output(inputs=input_perturbed, **model_params, **scalarize_params)

        # Iterate over units of interest
        for i in range(len(idx_interest)):

            # 8) Extract scalarized outputs for this unit of interest
            # Get indices of subsets associated with this unit of interest from the list of all subsets
            idx_excl_interest = []
            for subset_excl_interest in subsets_replace_excl_interest[i]:
                idx_excl_interest += [idx for (idx, subset) in enumerate(subsets_replace_all) if subset == subset_excl_interest]
            idx_incl_interest = []
            for subset_incl_interest in subsets_replace_incl_interest[i]:
                idx_incl_interest += [idx for (idx, subset) in enumerate(subsets_replace_all) if subset == subset_incl_interest]

            if isinstance(self.scalarized_model, TextScalarizedModel):
                # Iterate over similarity scores
                for key in self.scalarized_model.sim_scores:
                    # Extract scalarized output corresponding to original input/empty subset
                    scalar_output_orig = scalar_outputs[key][0].item()
                    # Extract scalarized outputs for this unit of interest
                    scalar_outputs_excl_interest = scalar_outputs[key][idx_excl_interest].cpu().numpy()
                    scalar_outputs_incl_interest = scalar_outputs[key][idx_incl_interest].cpu().numpy()
                    # Prepend output corresponding to empty subset
                    scalar_outputs_excl_interest = np.append(scalar_output_orig, scalar_outputs_excl_interest)

                    # 9) Compute Shapley values
                    normalization = get_normalization_constants(len(idx_replace_i[i]), max_units_replace) * (max_units_replace + 1)
                    importance_scores[key][idx_interest[i]] = np.inner(scalar_outputs_excl_interest - scalar_outputs_incl_interest, 1 / normalization)

            else:
                # Extract scalarized output(s) corresponding to original input/empty subset
                scalar_output_orig = scalar_outputs[[0]].cpu().numpy()
                # Extract scalarized outputs for this unit of interest
                scalar_outputs_excl_interest = scalar_outputs[idx_excl_interest].cpu().numpy()
                scalar_outputs_incl_interest = scalar_outputs[idx_incl_interest].cpu().numpy()
                # Prepend output corresponding to empty subset
                scalar_outputs_excl_interest = np.append(scalar_output_orig, scalar_outputs_excl_interest, axis=0)

                # 9) Compute Shapley values
                normalization = get_normalization_constants(len(idx_replace_i[i]), max_units_replace) * (max_units_replace + 1)
                importance_scores[idx_interest[i]] = np.dot(1 / normalization, scalar_outputs_excl_interest - scalar_outputs_incl_interest)

        # 10) Construct output dictionary
        if type(importance_scores) is not dict:
            # Convert importance_scores to dictionary
            if num_output_units == 1:
                importance_scores = importance_scores.squeeze(axis=1)
            if isinstance(self.scalarized_model, ProbScalarizedModel):
                # Label scores with type of scalarizer
                importance_scores = {"prob": importance_scores}
            else:
                importance_scores = {"score": importance_scores}
        # Add items to importance_scores dictionary
        importance_scores["units"] = input_orig
        importance_scores["unit_types"] = unit_types
        # Output dictionary
        output_dict = {"attributions": importance_scores, "output_orig": output_orig}

        return output_dict


def adapt_replacement_set(idx_replace, idx_interest, num_neighbors):
    """
    Adapt set of units that can be replaced to the unit of interest.

    This function modifies the indices of units that can be replaced to exclude the unit of interest
    and include neighbors within a specified range on either side.

    Args:
        idx_replace (np.ndarray of dtype int):
            Indices of units that can be replaced.
        idx_interest (int):
            Index of the unit of interest.
        num_neighbors (int):
            Number of neighbors on either side of the unit of interest to include.

    Returns:
        idx_replace_adapted (np.ndarray of dtype int):
            Adapted version of idx_replace, excluding the unit of interest and including neighbors.
    """
    # Location of idx_interest within idx_replace
    i_interest = np.nonzero(idx_replace == idx_interest)[0][0]
    # Lower and upper bounds of slice from idx_replace
    i_lower = max(i_interest - num_neighbors, 0)
    i_upper = i_interest + num_neighbors
    # Slice idx_replace
    idx_replace_adapted = idx_replace[i_lower : i_upper + 1]
    # Exclude unit of interest
    idx_replace_adapted = np.setdiff1d(idx_replace_adapted, idx_interest)

    return idx_replace_adapted

def get_normalization_constants(num_can_replace, max_units_replace):
    """
    Computes normalization constants for Shapley value calculation.

    Args:
        num_can_replace (int):
            The total number of units that can be replaced.
        max_units_replace (int):
            The maximum number of units that can be replaced at one time.

    Returns:
        normalization (np.ndarray):
            An array of normalization constants.
    """
    normalization = np.array([])
    for i in range(0, min(max_units_replace, num_can_replace) + 1):
        num_comb = comb(num_can_replace, i)
        normalization = np.hstack((normalization, np.repeat([num_comb], num_comb)))

    return normalization
