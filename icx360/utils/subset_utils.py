"""
Utilities that deal with subsets of input units.

These utilities are used by MExGen C-LIME (icx360.algorithms.mexgen.clime)
and L-SHAP (icx360.algorithms.mexgen.lshap).
"""
# Assisted by watsonx Code Assistant in formatting and augmenting docstrings.

from itertools import combinations
from math import ceil, inf
from random import sample

import numpy as np


def sample_subsets(idx_replace, max_units_replace, oversampling_factor=None, num_return_sequences=None,
                   empty_subset=False, return_weights=False):
    """
    Sample subsets of input units that can be replaced.

    Args:
        idx_replace ((num_replace,) np.ndarray):
            Indices of units that can be replaced.
        max_units_replace (int):
            Maximum number of units to replace at one time.
        oversampling_factor (float or None):
            Ratio of number of perturbed inputs to be generated to number of units that can be replaced.
            Default None means no upper bound on this ratio.
        num_return_sequences (int or None):
            Number of perturbed inputs to generate for each subset of units to replace.
        empty_subset (bool):
            Whether to include the empty subset.
        return_weights (bool):
            Whether to return weights associated with subsets.

    Returns:
        subsets (list[list[int]]):
            A list of subsets, where each subset is a list of unit indices.
        weights (list[float]):
            Weights associated with subsets, only returned if return_weights==True.
    """
    # Number of units that can be replaced
    num_replace = len(idx_replace)

    # Number of subsets to sample
    if oversampling_factor is not None and num_return_sequences is not None:
        num_subsets_remaining = ceil(oversampling_factor * num_replace / num_return_sequences)
    else:
        num_subsets_remaining = inf
    # Weight given to each subset size
    weight_k = num_subsets_remaining / (max_units_replace + empty_subset)

    # Initialize
    if empty_subset:
        subsets, weights = [[]], [weight_k]
    else:
        subsets, weights = [], []

    # Iterate over subset sizes
    for k in range(1, min(max_units_replace, num_replace) + 1):
        # Number of subsets of this size
        num_subsets_k = round(num_subsets_remaining / (max_units_replace + 1 - k)) if num_subsets_remaining < inf else inf

        # Enumerate subsets of size k
        # NOTE: Assumes that enumeration is reasonable for typical k <= 3 and num_replace < 100
        subsets_new = np.array(list(combinations(range(num_replace), k)))
        num_subsets_new = len(subsets_new)

        if num_subsets_new > num_subsets_k:
            # Subsample subsets to equal number specified for this size
            subsets_new = subsets_new[sample(range(num_subsets_new), num_subsets_k)]
            num_subsets_new = len(subsets_new)

        # Convert to subsets of unit indices
        subsets_new = idx_replace[subsets_new]

        # Add to subsets and update number remaining
        subsets.extend(subsets_new.tolist())
        weights.extend([weight_k / num_subsets_new] * num_subsets_new)
        num_subsets_remaining -= num_subsets_new
        if num_subsets_remaining <= 0:
            break

    if return_weights:
        return subsets, weights
    else:
        return subsets

def mask_subsets(units, subsets_replace, replacement_str):
    """
    Mask subsets of units with a fixed replacement string.

    Args:
        units (List[str]):
            Original sequence of units.
        subsets_replace (List[List[int]]):
            A list of subsets to replace, where each subset is a list of unit indices.
        replacement_str (str):
            String to replace units with (default "" for dropping units).

    Returns:
        input_masked (List[List[str]]):
            A list of masked versions of `units`,
            where each masked version corresponds to a subset in `subsets_replace`.
    """
    units = np.array(units)
    input_masked = []

    # Iterate over subsets of units
    for subset_replace in subsets_replace:
        # Replace units in subset with fixed replacement string
        units_masked = units.copy()
        units_masked[subset_replace] = replacement_str
        units_masked = units_masked.tolist()
        input_masked.append(units_masked)

    return input_masked
