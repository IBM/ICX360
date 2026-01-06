"""
Other utilities related to segmenting text into units.
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
        if unit_types[u] != "n" and not isalnum_string(unit):
            unit_types[u] = "n"

    return unit_types

def isalnum_string(string):
    # Check whether string contains any alphanumeric characters
    return any(c.isalnum() for c in string)

def merge_non_alphanumeric(units):
    """
    Merge non-alphanumeric units into adjacent units.

    Args:
        units (List[str]):
            List of units.

    Returns:
        units_merged (List[str]):
            List of units with non-alphanumeric units merged.
    """
    units_merged = []
    for unit in units:
        if units_merged and (not isalnum_string(unit) or not isalnum_string(units_merged[-1])):
            # Current unit or previous unit is not alphanumeric, merge with previous unit
            units_merged[-1] += unit
        else:
            units_merged.append(unit)

    return units_merged

def find_unit_boundaries(units, tokens):
    """
    Find boundaries of units in terms of tokens (starting token index of unit to starting index of next unit).

    Args:
        units (str or List[str]):
            List of units (or single unit if string).
        tokens (List[str]):
            List of tokens.

    Returns:
        boundaries (List[int]):
            A list of (num_units + 1) token indices, where boundaries[i]:boundaries[i+1] are the boundaries of unit i.
    """
    boundaries = [0]

    if type(units) is list and len(units) > 1:
        # More than one unit, find the ending index of each unit except the last
        idx_token = 0
        for unit in units[:-1]:
            # Look for the current token in the current unit
            token = tokens[idx_token].strip()
            idx_char = unit.find(token)
            # Stay in current unit if token found there,
            # or if not found and current unit is still long enough
            # The latter can happen with the second half of a token that is split between units
            # or special tokens that cannot be found anywhere in the text
            stay_in_unit = idx_char > -1 or len(unit) >= len(token)
            while stay_in_unit and idx_token < len(tokens) - 1:
                # Token found or skipped, advance to next token
                idx_token += 1
                if idx_char > -1:
                    # Token found, advance in the unit as well
                    unit = unit[idx_char + len(token):]
                # Look for the next token in the current unit
                token = tokens[idx_token].strip()
                idx_char = unit.find(token)
                stay_in_unit = idx_char > -1 or len(unit) >= len(token)
            # Token not found, record ending index of unit
            boundaries.append(idx_token)

    # Ending index of last unit
    boundaries.append(len(tokens))

    return boundaries
