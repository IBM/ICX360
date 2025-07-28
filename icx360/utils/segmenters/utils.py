"""
Other utilities for segmenting input text into units.
"""
# Assisted by watsonx Code Assistant in formatting and augmenting docstrings.

def exclude_non_alphanumeric(unit_types, units):
    """
    Exclude units without alphanumeric characters.

    Modifies the `unit_types` list by setting the type of units without alphanumeric characters to "n".

    Args:
        unit_types (list[str]):
            Types of units.
        units (list[str]):
            Sequence of units.

    Returns:
        unit_types (list[str]):
            Updated types of units.
    """
    # Check whether units that can be replaced have alphanumeric characters
    for u, unit in enumerate(units):
        if unit_types[u] != "n" and not any(c.isalnum() for c in unit):
            unit_types[u] = "n"

    return unit_types
