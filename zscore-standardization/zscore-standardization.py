import numpy as np


def zscore_standardize(
    X: list | np.ndarray, axis: int = 0, eps: float = 1e-12
) -> np.ndarray:
    """Returns population Z-scores mapping slices with std <= eps to zero."""
    new_x = np.asarray(X, dtype=float)

    mean = np.mean(new_x, axis=axis, keepdims=True)
    std = np.std(new_x, axis=axis, keepdims=True, ddof=0)

    constant_mask = std <= eps

    safe_std = np.where(constant_mask, 1.0, std)
    z = (new_x - mean) / safe_std

    z = np.where(constant_mask, 0.0, z)

    return z