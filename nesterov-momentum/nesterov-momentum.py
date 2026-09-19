import numpy as np

def nesterov_momentum_step(w: list, v: list, grad: list, lr: float = 0.01, momentum: float = 0.9) -> dict:
    """
    Returns a dictionary with new_w and new_v.
    """
    # Write code here
    w_arr = np.asarray(w, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    g_arr = np.asarray(grad, dtype=float)
    new_v = (momentum * v_arr) + (lr * g_arr)
    new_w = (w - new_v)
    my_dict = {"new_v" : new_v,
               "new_w" : new_w}
    return my_dict
    pass