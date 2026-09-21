import numpy as np

def adamw_step(w: list, m: list, v: list, grad: list, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, weight_decay: float = 0.01, eps: float = 1e-8) -> dict:
    """
    Returns a dictionary with new_w, new_m, and new_v.
    """
    arr_w = np.asarray(w, dtype=float)
    arr_m = np.asarray(m, dtype=float)
    arr_v = np.asarray(v, dtype=float)
    arr_g = np.asarray(grad, dtype=float)


    new_m = (arr_m * beta1) + (1 - beta1) * arr_g
    new_v = (beta2 * arr_v) + (1 - beta2) *(arr_g ** 2)
    new_w = arr_w - lr * (new_m / np.sqrt(new_v + (eps))) - (lr * weight_decay * arr_w)

    return {"new_m" : new_m, 
            "new_v" : new_v, 
            "new_w" : new_w }

    


    # Write code here
    pass