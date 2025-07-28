"""File containing general utility functions
"""

import random

import numpy as np
import torch


def select_device():
    """Select device on which to perform all operations.

    Returns:
        device (str): device on which to perform all operations according
            to user system
    """

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    return device


def fix_seed(seed=12345):
    """
    Fix a random seeed for all random number generators (random, numpy, torch)

    Args:
        seed: seed to set for all randomizations
    """

    # Fix seed for experimentation
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.mps.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
