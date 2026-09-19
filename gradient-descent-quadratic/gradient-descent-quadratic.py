# 1. Define the function
def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Returns the final scalar x after the requested iterations.
    """
    x = x0
    for _ in range(steps):
        grad = 2 * a * x + b
        x = x - lr * grad
    return x

result = gradient_descent_quadratic(a=1.0, b=-4.0, c=3.0, x0=0.0, lr=0.1, steps=50)
print(result)