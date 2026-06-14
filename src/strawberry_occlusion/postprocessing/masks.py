"""Mask post-processing helpers."""

import numpy as np


def threshold_mask(probability_map: np.ndarray, threshold: float) -> np.ndarray:
    """Convert a probability map into a boolean mask using a configured threshold."""

    return np.asarray(probability_map) >= threshold
