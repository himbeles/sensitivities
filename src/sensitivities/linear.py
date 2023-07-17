import numpy as np
import numpy.typing as npt
from scipy.optimize import approx_fprime

_epsilon = np.sqrt(np.finfo(float).eps)

def propagate_errors(
    f,
    errors: npt.ArrayLike,
    x0: npt.ArrayLike,
    epsilon: float | npt.ArrayLike = _epsilon,
) -> np.ndarray:

    """
    Propagate errors of function input arguments 
    based on differentials around central values.

    It uses the scipy function `approx_fprime` to calculate the 
    partial derivatives of f. The propagated error is calculated 
    by taking the square root of the sum of squares of the product 
    of the partials and the uncertainties.

    Args:
        f (function): The function for which the propagation of errors 
            is calculated.
        errors: Uncertainties in the arguments of f. 
            Each uncertainty corresponds to a single argument of f.
        x0: Initial values for the arguments of f. 
            The function f will be evaluated around these points.
        epsilon (optional): Step sizes for the central difference approximation
            of the gradient. Defaults to sqrt(machine epsilon) 
            for float data type. If it is an array, 
            it should have the same shape as x0.

    Returns:
        Propagated errors. Each element is the propagated error for a parameter of f 
                    evaluated at the corresponding value in x0.

    Example:
        >>> def my_function(x, y):
        ...     return [x ** 2, y]

        >>> propagate_errors(my_function, errors=[0.1, 0.2], x0=[1, 1])
        np.array([0.2, 0.2])
    """

    # Wrapper function to allow f to accept a single list of parameters
    def wrapper(x):
        return f(*x)

    # Calculate partial derivatives
    partials = approx_fprime(x0, wrapper, epsilon)

    # Calculate error in function output
    errors = np.array(errors)
    propagated_errors = np.sqrt(np.sum(np.square(partials * errors), axis=-1))

    return propagated_errors
