"""
shared/vector.py
----------------
Tiny 2D vector helpers used across both algorithms.
We keep them as plain numpy arrays (shape: (2,)) for simplicity.
"""

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    """
    Return a unit vector in the same direction as v.

    Hint:
      - compute the magnitude (np.linalg.norm)
      - if magnitude == 0, return v unchanged (avoid division by zero)
      - otherwise return v / magnitude
    """
    magnitude = np.linalg.norm(v)
    if magnitude == 0:
        return v
    return v / magnitude


def limit(v: np.ndarray, max_magnitude: float) -> np.ndarray:
    """
    Clamp the magnitude of v to max_magnitude.
    If |v| <= max_magnitude, return v unchanged.

    Hint:
      - reuse normalize() — scale the unit vector by max_magnitude
    """
    magnitude = np.linalg.norm(v)
    if magnitude <= max_magnitude:
        return v
    return normalize(v) * max_magnitude    


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Euclidean distance between two 2D points.
    """
    return np.linalg.norm(a - b)
   