"""
Utilities for coloring and displaying units of text.
"""

import matplotlib.colors as mcolors
import numpy as np
from IPython.display import HTML, display

COLOR_LIST_IBM_30 = ["#a6c8ff", "#c6c6c6", "#ffb3b8"]
COLOR_LIST_IBM_40 = ["#78a9ff", "#c6c6c6", "#ff8389"]


def highlight_text(unit, color):
    return f'<span style="background-color: {color}; padding: 2px;">{unit}</span>'

def color_units(units, scores, norm_factor=None, scale_sqrt=True, color_list=COLOR_LIST_IBM_40, show=True):
    """
    Color units of text according to scores and display.

    Args:
        units ((num_units,) np.ndarray):
            Units of text.
        scores ((num_units,) np.ndarray):
            Scores corresponding to units.
        norm_factor (float or None):
            Factor to divide scores by to normalize them. None (default) means np.abs(scores).max().
        scale_sqrt (bool):
            Whether to apply square root to magnitude of score
        color_list (List[str]):
            List of colors for matplotlib.colors.LinearSegmentedColormap
        show (bool):
            Show on screen if True, otherwise return list of HTML strings.

    Returns:
        colored_units (List[str] or None):
            List of HTML-formatted units of text if show==False, otherwise None.
    """
    # Normalize scores by dividing by norm_factor
    if norm_factor is None:
        norm_factor = np.abs(scores).max()
    scores_norm = scores / norm_factor if norm_factor else scores.copy()

    if scale_sqrt:
        # Apply square root to magnitude
        sqrt_mag = lambda x: np.sign(x) * (np.abs(x) ** .5)
        scores_norm = sqrt_mag(scores_norm)

    # Map scores to [0, 1]
    normalize = mcolors.Normalize(vmin=-1, vmax=1)
    scores_norm = normalize(scores_norm)

    # Colormap
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "blue_to_red", color_list
    )

    colored_units = []
    # Iterate over units
    for u in range(len(units)):
        # Map score to color and highlight corresponding text
        color = mcolors.to_hex(cmap(scores_norm[u]))
        colored_unit = highlight_text(units[u], color)
        colored_units.append(colored_unit)

    if show:
        return display(HTML(" ".join(colored_units)))
    else:
        return colored_units
