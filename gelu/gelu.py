import math
import numpy as np


def gelu(x: list) -> np.ndarray:
    """Computes exact GELU activation using only standard math and NumPy."""
    x_arr = np.asarray(x, dtype=float)

    # Vectorize math.erf to work over NumPy arrays
    erf_vec = np.vectorize(math.erf)

    # GELU(x) = (x / 2) * (1 + erf(x / sqrt(2)))
    return (x_arr / 2.0) * (1.0 + erf_vec(x_arr / np.sqrt(2.0)))