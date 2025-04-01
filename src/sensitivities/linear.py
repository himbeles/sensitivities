import numpy as np
import numpy.typing as npt
from scipy.optimize import approx_fprime

from .math import is_valid_correlation_matrix

_epsilon = np.sqrt(np.finfo(float).eps)


def propagate_errors(
    f,
    errors: npt.ArrayLike,
    x0: npt.ArrayLike,
    corr: npt.ArrayLike | None = None,
    epsilon: float | npt.ArrayLike = _epsilon,
) -> np.ndarray:
    """
    Propagate errors of function input arguments
    based on differentials around central values.

    It uses the scipy function `approx_fprime` to calculate the
    partial derivatives of f. The propagated error is calculated
    by taking the square root of the sum of squares of the product
    of the partials and the uncertainties.

    If a correlation matrix is provided, the propagated error
    is calculated as the square root of the sum of squares of the
    product of the partials, the correlation matrix, and the
    uncertainties.

    Args:
        f (function): The function for which the propagation of errors
            is calculated.
        errors: Uncertainties in the arguments of f.
            Each uncertainty corresponds to a single argument of f.
        x0: Center values for the arguments of f.
            The function f will be evaluated around these points.
        corr (optional): Correlation matrix of the uncertainties.
            If provided, it should be a square matrix with dimensions
            equal to the length of errors. The correlation matrix must
            be symmetric and positive semi-definite.
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

        >>> corr = np.array([[1, 1], [1, 1]])
        >>> propagate_errors(my_function, errors=[0.1, 0.2], x0=[1, 1], corr=corr)
        np.array([0.3, 0.3])
    """

    errors = np.array(errors)

    # Wrapper function to allow f to accept a single list of parameters
    def wrapper(x):
        return f(*x)

    # Calculate partial derivatives
    partials = approx_fprime(x0, wrapper, epsilon)

    if corr is not None:
        corr = np.array(corr)
        if corr.shape != (len(errors), len(errors)):
            raise ValueError(
                "Correlation matrix must be square with dimensions equal to the length of errors."
            )
        # assert correlation matrix is symmetric and positive definite
        if not is_valid_correlation_matrix(corr):
            raise ValueError(
                "Correlation matrix must be symmetric and positive semi-definite."
            )

        propagated_errors = np.sqrt(
            np.einsum(
                "...i,...ij,...j->...", errors * partials, corr, errors * partials
            )
        )

    else:
        propagated_errors = np.sqrt(np.sum(np.square(partials * errors), axis=-1))

    return propagated_errors
