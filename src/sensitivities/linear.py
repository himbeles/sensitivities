import numpy as np
import numpy.typing as npt
from scipy.optimize import approx_fprime

from .math import (
    correlation_to_covariance,
    covariance_to_correlation,
    is_valid_correlation_matrix,
)

_epsilon = np.sqrt(np.finfo(float).eps)


def propagate_uncertainties(
    f,
    uncertainties: npt.ArrayLike,
    x0: npt.ArrayLike,
    corr: npt.ArrayLike | None = None,
    epsilon: float | npt.ArrayLike = _epsilon,
) -> np.ndarray:
    """
    Propagate uncertainties of function input arguments
    based on differentials around central values.
    The propagated uncertainty is calculated
    as in the BIPM GUM (JCGM 102:2011).

    Handles univariate and multivariate measurement functions f
    (functions with one or more **output** variables).

    It uses the scipy function `approx_fprime` to calculate the
    partial derivatives of f.

    Args:
        f (function): The function for which the propagation of uncertainties
            is calculated; can be multivariate.
        uncertainties: Uncertainties in the arguments of f.
            Each uncertainty corresponds to a single argument of f.
        x0: Center values for the arguments of f.
            The function f will be evaluated around these points.
        corr (optional): Correlation matrix of the uncertainties.
            If provided, it should be a square matrix with dimensions
            equal to the length of uncertainties. The correlation matrix must
            be symmetric and positive semi-definite.
        epsilon (optional): Step sizes for the central difference approximation
            of the gradient. Defaults to sqrt(machine epsilon)
            for float data type. If it is an array,
            it should have the same shape as x0.

    Returns:
        Propagated uncertainties. Each element is the propagated uncertainty
            for a parameter of f evaluated at the corresponding value in x0.

    Example:
        >>> def my_function(x, y):
        ...     return [x ** 2, y]

        >>> propagate_uncertainties(my_function, uncertainties=[0.1, 0.2], x0=[1, 1])
        np.array([0.2, 0.2])

        >>> corr = np.array([[1, 1], [1, 1]])
        >>> propagate_uncertainties(my_function, uncertainties=[0.1, 0.2], x0=[1, 1], corr=corr)
        np.array([0.3, 0.3])
    """

    uncertainties = np.array(uncertainties)

    # Wrapper function to allow f to accept a single list of parameters
    def wrapper(x):
        return f(*x)

    # Calculate partial derivatives around x0
    partials = approx_fprime(x0, wrapper, epsilon)

    if corr is not None:
        corr = np.array(corr)
        if corr.shape != (len(uncertainties), len(uncertainties)):
            raise ValueError(
                "Correlation matrix must be square with dimensions equal to the length of uncertainties."
            )
        # assert correlation matrix is symmetric and positive definite
        if not is_valid_correlation_matrix(corr):
            raise ValueError(
                "Correlation matrix must be symmetric and positive semi-definite."
            )

    else:
        corr = np.eye(len(uncertainties))

    # error propagation as in BIPM GUM (JCGM 102:2011)
    cov = correlation_to_covariance(corr, uncertainties)
    partials_2d = np.atleast_2d(partials)
    f_cov = partials_2d @ cov @ np.permute_dims(partials_2d, [-1, -2])
    f_corr, f_uncertainties = covariance_to_correlation(f_cov)

    if np.shape(partials_2d)[0] > 1:
        return f_uncertainties, f_corr
    else:
        return f_uncertainties[0]
