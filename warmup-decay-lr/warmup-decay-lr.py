import numpy as np
def warmup_decay_schedule(base_lr: float, warmup_steps: int, total_steps: int, current_step: int) -> float:
    """
    Returns the learning rate for the requested training step.
    """
    # Write code here
    if current_step < warmup_steps:
        lr = base_lr * (current_step / warmup_steps)

    else:
        decay_steps = total_steps - warmup_steps
        steps_left = total_steps - current_step
        lr = base_lr * (steps_left / decay_steps)

    return float(lr)