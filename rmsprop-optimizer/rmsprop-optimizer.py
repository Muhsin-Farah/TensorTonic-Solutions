import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Returns (new_w, new_s) with the same shapes as the inputs.
    """
    # Write code here
    new_w = np.asarray(w, dtype=float)
    new_g = np.asarray(g, dtype=float)
    new_s = np.asarray(s, dtype=float)

    average = (beta * new_s) + (1 - beta) * (new_g ** 2)
    update = new_w - (lr / (np.sqrt(average) + eps)) * new_g
    return update.tolist(), average.tolist()
    
    pass