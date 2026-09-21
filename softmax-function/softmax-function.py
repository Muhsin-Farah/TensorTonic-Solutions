import numpy as np


def softmax(x: list) -> np.ndarray:
    """Returns stable softmax probabilities as a NumPy array matching the shape of x."""
    new_x = np.asarray(x, dtype=float)

    # 1. Compute max along the last dimension for numerical stability
    m = np.max(new_x, axis=-1, keepdims=True)

    # 2. Subtract max inside exp
    exp_x = np.exp(new_x - m)

    # 3. Normalize along the last dimension
    probs = exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    return probs