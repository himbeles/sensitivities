import numpy as np
import numpy.typing as npt
from scipy.optimize import approx_fprime

_epsilon = np.sqrt(np.finfo(float).eps)

def propagate_errors(
    f,
    errors: npt.ArrayLike,
    x0: npt.ArrayLike,
    epsilon: float | npt.ArrayLike = _epsilon,
):
    # Wrapper function to allow f to accept a single list of parameters
    def wrapper(x):
        return f(*x)

    # Calculate partial derivatives
    partials = approx_fprime(x0, wrapper, epsilon)

    # Calculate error in function output
    propagated_errors = np.sqrt(np.sum(np.square(partials * errors)))

    return propagated_errors
