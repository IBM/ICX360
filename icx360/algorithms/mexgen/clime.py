"""
Class and supporting functions for MExGen C-LIME explainer.

The MExGen framework and C-LIME algorithm are described in:
    Multi-Level Explanations for Generative Language Models.
    Lucas Monteiro Paes and Dennis Wei et al.
    The 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025).
    https://arxiv.org/abs/2403.14459
"""
# Assisted by watsonx Code Assistant in formatting and augmenting docstrings.

import numpy as np
from sklearn.linear_model import LinearRegression, lars_path

from icx360.algorithms.lbbe import LocalBBExplainer
from icx360.utils.scalarizers import ProbScalarizedModel, TextScalarizedModel
from icx360.utils.segmenters import SpaCySegmenter, exclude_non_alphanumeric
from icx360.utils.subset_utils import mask_subsets, sample_subsets


class CLIME(LocalBBExplainer):
    """
    MExGen C-LIME explainer

    Attributes:
        model (icx360.utils.model_wrappers.Model):
            Model to explain, wrapped in an icx360.utils.model_wrappers.Model object.
        segmenter (icx360.utils.segmenters.SpaCySegmenter):
            Object for segmenting input text into units using a spaCy model.
        scalarized_model (icx360.utils.scalarizers.Scalarizer):
            "Scalarized model" that further wraps `model` with a method for computing scalar values
            based on the model's inputs or outputs.
    """
    def __init__(self, model, segmenter="en_core_web_trf", scalarizer="prob", **kwargs):
        """
        Initialize MExGen C-LIME explainer.

        Args:
            model (icx360.utils.model_wrappers.Model):
                Model to explain, wrapped in an icx360.utils.model_wrappers.Model object.
            segmenter (str):
                Name of spaCy model to use in segmenter (icx360.utils.segmenters.SpaCySegmenter).
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

        # Instantiate segmenter
        self.segmenter = SpaCySegmenter(segmenter)

        # Instantiate scalarized model
        if scalarizer == "prob":
            self.scalarized_model = ProbScalarizedModel(model)
        elif scalarizer == "text":
            self.scalarized_model = TextScalarizedModel(model, **kwargs)
        else:
            raise ValueError("Scalarizer not supported")

    def explain_instance(self, input_orig, unit_types="p", ind_segment=True, segment_type="s", max_phrase_length=10,
                         model_params={}, scalarize_params={}, oversampling_factor=10, max_units_replace=2,
                         empty_subset=True, replacement_str="", num_nonzeros=None, debias=True):
        """
        Explain model output by attributing it to parts of the input text.

        Uses an algorithm called C-LIME (a variant of LIME)
        to fit a local linear approximation to the model and compute attribution scores.

        Args:
            input_orig (str or List[str]):
                [input] Input text as a single unit (if str) or segmented sequence of units (List[str]).
            unit_types (str or List[str]):
                [input] Types of units in input_orig.
                    "p" for paragraph, "s" for sentence, "w" for word,
                    "n" for not to be perturbed/attributed to.
                If str, applies to all units in input_orig, otherwise unit-specific.
            ind_segment (bool or List[bool]):
                [segmentation] Whether to segment input text.
                If bool, applies to all units; if List[bool], applies to each unit individually.
            segment_type (str):
                [segmentation] Type of units to segment into: "s" for sentences, "w" for words, "ph" for phrases.
            max_phrase_length (int):
                [segmentation] Maximum phrase length in terms of spaCy tokens (default 10).
            model_params (dict):
                Additional keyword arguments for model generation (for the self.model.generate() method).
            scalarize_params (dict):
                Additional keyword arguments for computing scalar outputs
                (for the self.scalarized_model.scalarize_output() method).
            oversampling_factor (float):
                [perturbation] Ratio of number of perturbed inputs to be generated to number of units that can be perturbed.
            max_units_replace (int):
                [perturbation] Maximum number of units to perturb at one time (default 2).
            empty_subset (bool):
                [perturbation] Whether to include empty subset of units to perturb (default True).
            replacement_str (str):
                [perturbation] String to replace units with (default "" for dropping units).
            num_nonzeros (int or None):
                [linear model] Number of non-zero coefficients in linear model (default None means dense model).
            debias (bool):
                [linear model] Refit linear model with no penalty after selecting features (default True).

        Returns:
            output_dict (dict):
                Dictionary with the following items:
                    "attributions" (dict):
                        Dictionary with attribution scores, corresponding input units, and unit types.
                    "output_orig" (icx360.utils.model_wrappers.GeneratedOutput):
                        Output object generated from original input.
                    "intercept" (float or dict[float]):
                        Intercept(s) of linear model.

                Items in "attributions" dictionary:
                    "units" (List[str]):
                        input_orig segmented into units if not already, otherwise same as original.
                    "unit_types" (List[str]):
                        Types of units.
                    `score_label` ((num_units,) np.ndarray):
                        One or more sets of attribution scores (labelled by the type of scalarizer).
        """
        # 1) Segment input text if needed
        if type(ind_segment) is bool:
            ind_segment = [ind_segment]
        if type(input_orig) is str or any(ind_segment):
            # Call segmenter
            input_orig, unit_types, _ = self.segmenter.segment_units(input_orig, ind_segment, unit_types, segment_type=segment_type, max_phrase_length=max_phrase_length)
            # Exclude units without alphanumeric characters from perturbation
            unit_types = exclude_non_alphanumeric(unit_types, input_orig)
        num_units = len(input_orig)

        if type(unit_types) is str:
            # Expand to list
            unit_types = [unit_types] * num_units

        # 2) Generate output for original input
        output_orig = self.model.generate([input_orig], text_only=False, **model_params)

        # 3) Enumerate subsets of units that will be perturbed/replaced
        idx_replace = (np.array(unit_types) != "n").nonzero()[0]
        subsets_replace, subset_weights = sample_subsets(idx_replace, max_units_replace, oversampling_factor, num_return_sequences=1, empty_subset=empty_subset, return_weights=True)

        # 4) Replace subsets of units with replacement string
        input_perturbed = mask_subsets(input_orig, subsets_replace, replacement_str)

        # 5) Compute scalarized outputs for perturbed inputs to provide targets for linear model
        if "ref_output" not in scalarize_params:
            # Reference output is original output if not specified
            scalarize_params = scalarize_params.copy()
            scalarize_params["ref_output"] = output_orig

        target = self.scalarized_model.scalarize_output(inputs=input_perturbed, **model_params, **scalarize_params)

        # 6) Features for linear model
        features = compute_linear_model_features(subsets_replace, num_units)

        # 7) Fit linear model
        if type(target) is dict:
            # Iterate over multiple target vectors
            coef = {}
            intercept = {}
            num_nonzeros_out = {}
            for key in target.keys():
                coef[key], intercept[key], num_nonzeros_out[key] = fit_linear_model(features, target[key].cpu().numpy(), subset_weights, num_nonzeros, debias)

        else:
            # Single target vector
            coef, intercept, num_nonzeros_out = fit_linear_model(features, target.cpu().numpy(), subset_weights, num_nonzeros, debias)

        # 8) Construct output dictionary
        if type(coef) is not dict:
            # Convert coef to dictionary
            if isinstance(self.scalarized_model, ProbScalarizedModel):
                # Label scores with type of scalarizer
                coef = {"prob": coef}
            else:
                coef = {"score": coef}
        # Add items to coef dictionary
        coef["units"] = input_orig
        coef["unit_types"] = unit_types
        # Output dictionary
        output_dict = {"attributions": coef, "output_orig": output_orig, "intercept": intercept}

        return output_dict


def compute_linear_model_features(subsets_replace, num_units):
    """
    Compute features used by explanatory linear model.

    This function generates a feature matrix for a linear model that explains
    the impact of perturbing specific input units.

    Args:
        subsets_replace (List[List[int]]):
            A list of subsets, where each subset is a list of indices
            corresponding to the units that have been replaced.
        num_units (int):
            Total number of units.

    Returns:
        features ((num_perturb, num_units) np.ndarray):
            Matrix of feature values,
            equal to 1 if the unit is part of the perturbed subset, and 0 otherwise.
    """
    num_perturb = len(subsets_replace)
    features = np.zeros((num_perturb, num_units))

    # Iterate over subsets of units
    for s, subset_replace in enumerate(subsets_replace):
        # Set feature values corresponding to replaced units
        features[s, subset_replace] = 1.

    return features

def fit_linear_model(features, target, sample_weights, num_nonzeros, debias):
    """
    Fit explanatory linear model.

    Args:
        features ((num_perturb, num_units) np.ndarray):
            Feature values.
        target ((num_perturb,) np.ndarray):
            Target values to predict.
        sample_weights ((num_perturb,) np.ndarray):
            Sample weights.
        num_nonzeros (int or None):
            Number of non-zero coefficients desired in linear model, None means dense model.
        debias (bool):
            Refit linear model with no penalty after selecting features.

    Returns:
        coef ((num_units,) np.ndarray):
            Coefficients of linear model.
        intercept (float):
            Intercept of linear model.
        num_nonzeros (int):
            Actual number of non-zero coefficients.
    """
    num_units = features.shape[1]

    if num_nonzeros is None:
        # Fit dense linear model over the units that were perturbed (`active`)
        active = features.any(axis=0).nonzero()[0]
        coef = np.zeros(num_units)
        lr = LinearRegression()
        lr.fit(features[:, active], target, sample_weight=sample_weights)
        coef[active] = lr.coef_
        intercept = lr.intercept_

    else:
        # Fit sparse linear model

        # Center feature and target values
        features_mean = features.mean(axis=0)
        target_mean = target.mean()
        features_centered = features - features_mean
        target_centered = target - target_mean

        # Call lars_path to obtain sparse linear model with num_nonzeros coefficients
        # NOTE: may return fewer than num_nonzeros if coefficients leave the active set
        alphas, active, coef = lars_path(np.sqrt(sample_weights)[:, None] * features_centered, np.sqrt(sample_weights) * target_centered, max_iter=num_nonzeros, method="lasso", return_path=False)

        if debias:
            coef = np.zeros(num_units)
            if len(active):
                # Refit linear model on selected features with no penalty
                lr = LinearRegression()
                lr.fit(features[:, active], target, sample_weight=sample_weights)
                coef[active] = lr.coef_
                intercept = lr.intercept_
            else:
                # No active set, coefficients all zero
                intercept = target_mean
        else:
            # Compute intercept to account for centering
            intercept = target_mean - coef @ features_mean

    # Negate coefficients so that important units have positive coefficients
    return -coef, intercept, len(active)
