import numpy as np

def hinge_loss(y_true: list, y_score: list, margin: float = 1.0, reduction: str = "mean") -> float:
    """
    Returns the loss as a float.
    """
    # Write code here
    new_yt = np.asarray(y_true, dtype=float)
    new_ys = np.asarray(y_score, dtype=float)

    sample_loss = np.maximum(0, (margin - new_yt * new_ys))
    if reduction == "mean":
        return float(np.mean(sample_loss))
    elif reduction == "sum":
        return float(np.sum(sample_loss))
    else:
        return sample_loss.tolist()
   
    return sample_loss
    pass