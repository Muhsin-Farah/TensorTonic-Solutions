import numpy as np

def mean_squared_error(y_pred: list, y_true: list) -> float:
    """
    Returns the error as a float.
    """
    pred = np.asarray(y_pred, dtype=float)
    other = np.asarray(y_true, dtype=float)
    squared = np.mean((pred-other) ** 2)
    return squared
    
    pass