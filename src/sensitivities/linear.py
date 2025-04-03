from inspect import getfullargspec
from typing import Union
import numpy as np
import numpy.typing as npt
from scipy.optimize import approx_fprime

from .distributions import (
    Distribution,
    Gaussian,
    Uniform,
    Discrete,
    Fixed,
    _DistributionOrValue,
)

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
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Propagate uncertainties of function input arguments
    based on differentials around central values.
    The propagated uncertainty and correlation is calculated
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
        Propagated correlation. If the function f is multivariate in its output,
            the correlation between the outputs variables is provided as a
            correlation matrix.

    Example:
        >>> def my_function(x, y):
        ...     return [x ** 2, y]

        >>> propagate_uncertainties(my_function, uncertainties=[0.1, 0.2], x0=[1, 1])
        np.array([0.2, 0.2])

        >>> corr = np.array([[1, 1], [1, 1]])
        >>> propagate_uncertainties(my_function, uncertainties=[0.1, 0.2], x0=[1, 1], corr=corr)
        np.array([0.3, 0.3])
    """

    uncertainties = np.asarray(uncertainties)

    # Wrapper function to allow f to accept a single list of parameters
    def wrapper(x):
        return f(*x)

    # Calculate partial derivatives around x0
    partials = np.asarray(approx_fprime(x0, wrapper, epsilon))

    if corr is not None:
        corr = np.asarray(corr)
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


def propagate_uncertainty_distributions(
    f,
    distributions_args: list[_DistributionOrValue] = [],
    distributions_kwargs: dict[str, _DistributionOrValue] = {},
    corr: Union[np.ndarray, list[tuple[str, str, float]], None] = None,
    epsilon: float | npt.ArrayLike = _epsilon,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Propagate uncertainties of function input arguments
    based on differentials around central values.
    The propagated uncertainty and correlation is calculated
    as in the BIPM GUM (JCGM 102:2011).

    `Distribution` objects can be given for each function input argument.
    Their associated standard deviation and mean are used
    as standard uncertainty and central values
    for the uncertainty propagation.

    Handles univariate and multivariate measurement functions f
    (functions with one or more **output** variables).

    It uses the scipy function `approx_fprime` to calculate the
    partial derivatives of f.

    Args:
        f (function): The function for which the propagation of uncertainties
            is calculated; can be multivariate.
        distributions_args: A list of Distribution instances or fixed values for each positional
            argument of the function. The order should match the order of the function's positional
            arguments. If a fixed value is provided, it will be treated as a Fixed Distribution.
            Defaults to an empty list.
        distributions_kwargs: A dictionary where the keys are the names of keyword arguments in the
            function, and the values are Distribution instances or fixed values for these keyword arguments.
            If a scalar value is provided, it will be treated as a Fixed Distribution.
            Defaults to an empty dictionary.
        corr: Optional correlation specification. Either:
            - A correlation matrix (numpy.ndarray) of shape (num_args, num_args), or
            - A list of tuples specifying correlations as (arg1, arg2, correlation_value).
        epsilon (optional): Step sizes for the central difference approximation
            of the gradient. Defaults to sqrt(machine epsilon)
            for float data type. If it is an array,
            it should have the same shape as x0.

    Returns:
        Propagated uncertainties. Each element is the propagated uncertainty
            for a parameter of f evaluated at the corresponding value in x0.
        Propagated correlation. If the function f is multivariate in its output,
            the correlation between the outputs variables is provided as a
            correlation matrix.

    Example:
        >>> def my_function(x, y):
        ...     return [x ** 2, y]

        >>> propagate_uncertainties(my_function, uncertainties=[0.1, 0.2], x0=[1, 1])
        np.array([0.2, 0.2])

        >>> corr = np.array([[1, 1], [1, 1]])
        >>> propagate_uncertainties(my_function, uncertainties=[0.1, 0.2], x0=[1, 1], corr=corr)
        np.array([0.3, 0.3])
    """

    def convert_to_dist(d):
        if isinstance(d, Distribution):
            return d
        elif isinstance(d, (int, float, np.ndarray)):
            return Fixed(d)
        else:
            raise TypeError("Unsupported type in distributions list")

    distributions_list = [convert_to_dist(d) for d in distributions_args]
    distributions_dict = {
        k: convert_to_dist(v) for k, v in distributions_kwargs.items()
    }
    all_distributions = distributions_list + list(distributions_dict.values())
    num_dist = len(all_distributions)

    argspec = getfullargspec(f)
    param_names = argspec.args
    num_positional_args = (
        len(argspec.args) - len(argspec.defaults)
        if argspec.defaults
        else len(argspec.args)
    )

    if len(distributions_list) != num_positional_args:
        raise ValueError(
            f"Number of distributions in list does not match number of positional parameters in function {f.__name__}"
        )

    if not set(distributions_dict.keys()).issubset(set(argspec.args)):
        raise ValueError(
            f"Some keys in distributions dictionary do not match parameter names in function {f.__name__}"
        )

    if corr is not None:
        if isinstance(corr, list):
            corr_matrix = np.eye(num_dist)
            param_to_idx = {name: idx for idx, name in enumerate(param_names)}
            for a, b, val in corr:
                i, j = param_to_idx[a], param_to_idx[b]
                corr_matrix[i, j] = corr_matrix[j, i] = val
        elif isinstance(corr, np.ndarray):
            if corr.shape != (num_dist, num_dist):
                raise ValueError(
                    "Correlation matrix must be square with dimensions equal to the length of uncertainties."
                )
            corr_matrix = corr
        else:
            raise TypeError(
                "corr must be either a correlation matrix or a list of tuples"
            )

    # mean and standard uncertainties for input distributions
    uncertainties = np.asarray([dist.std() for dist in all_distributions])
    x0 = np.asarray([dist.mean() for dist in all_distributions])

    # Wrapper function to allow f to accept a single list of parameters
    param_names_with_dists = param_names[:num_positional_args] + list(
        distributions_dict.keys()
    )

    def wrapper(*x):
        bound_args = dict(zip(param_names_with_dists, x))
        return f(**bound_args)

    return propagate_uncertainties(
        wrapper,
        uncertainties=uncertainties,
        x0=x0,
        corr=corr_matrix,
        epsilon=epsilon,
    )
