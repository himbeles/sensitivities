from inspect import getfullargspec
from typing import Callable, Dict, List, Sequence, Union

import numpy as np
import numpy.typing as npt
import scipy.stats

from .math import is_valid_correlation_matrix
from .distributions import (
    Distribution,
    Gaussian,
    Uniform,
    Discrete,
    Fixed,
    _DistributionOrValue,
)


def sample(
    f: Callable,
    distributions_args: list[_DistributionOrValue] = [],
    distributions_kwargs: dict[str, _DistributionOrValue] = {},
    corr: Union[np.ndarray, list[tuple[str, str, float]], None] = None,
    n: int = 100,
) -> List:
    """
    Sample output values from a function. Each function argument is stochastically
    sampled from a provided distribution, optionally considering correlations
        between input distributions.

    Args:
        f: The function to sample.
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
        n: The number of samples to draw from the function. Defaults to 100.

    Returns:
        A list of function return values for the stochastically varied inputs.

    Raises:
        TypeError: If an unsupported type is provided in the distributions list or dictionary.
        ValueError: If the number of distributions in the list does not match the number of
            positional parameters in the function, or if some keys in the distributions
            dictionary do not match parameter names in the function.
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

        samples = sample_distributions(all_distributions, n, corr_matrix)
        res_list = []
        for j in range(n):
            new_args = list(samples[j, : len(distributions_list)])
            new_kwargs = {
                k: samples[j, len(distributions_list) + i]
                for i, k in enumerate(distributions_dict.keys())
            }
            res_list.append(f(*new_args, **new_kwargs))
        return res_list

    else:
        res_list = []
        for j in range(n):
            new_args = [dist.sample() for dist in distributions_list]
            new_kwargs = {k: dist.sample() for k, dist in distributions_dict.items()}
            res_list.append(f(*new_args, **new_kwargs))
        return res_list


def sample_distributions(
    distributions: Sequence[Distribution],
    n: int = 100,
    corr: npt.ArrayLike | None = None,
    seed=None,
):
    """
    Sample values from a list of distributions.

    Args:
        distributions: A list of Distribution instances.
        n: The number of samples to draw from the distributions. Defaults to 100.
        corr: Optional correlation specification. A correlation matrix (numpy.ndarray) of shape
            (len(distributions), len(distributions)). Only Gaussian and Uniform distributions
            can be correlated.

    Returns:
        A list of samples from the distributions.
    """
    num_dist = len(distributions)
    samples = np.empty((n, num_dist))
    dist_ind_without_correlation = set(range(num_dist))

    if corr is not None and not np.allclose(corr, np.eye(num_dist, num_dist)):
        corr = np.asarray(corr)
        if corr.shape != (num_dist, num_dist):
            raise ValueError(
                "Correlation matrix must be square with dimensions equal to the length of uncertainties."
            )
        # assert correlation matrix is symmetric and positive definite
        if not is_valid_correlation_matrix(corr):
            raise ValueError(
                "Correlation matrix must be symmetric and positive semi-definite."
            )

        # check that correlation is only non-zero between uniform and gaussians distributions
        allowed_distributions = (Uniform, Gaussian)
        dist_ind_with_correlation = set()
        for i in range(num_dist):
            for j in range(i + 1, num_dist):
                if corr[i, j] != 0:
                    if not isinstance(
                        distributions[i], allowed_distributions
                    ) or not isinstance(distributions[j], allowed_distributions):
                        raise ValueError(
                            "Correlation can only be non-zero between uniform and gaussian distributions."
                        )
                    dist_ind_with_correlation.add(i)
                    dist_ind_with_correlation.add(j)

        # sub-correlation-matrix to be used for multivariate copula method
        dist_ind_without_correlation.difference_update(dist_ind_with_correlation)
        dist_ind_with_correlation = tuple(sorted(dist_ind_with_correlation))
        corr_red = corr[np.ix_(dist_ind_with_correlation, dist_ind_with_correlation)]
        n_red = len(corr_red)

        rng = np.random.default_rng(seed)

        # make basis multivariate normal distributions (mean=0, sigma=1)
        basis_normal = rng.multivariate_normal(
            mean=np.zeros(n_red), cov=corr_red, size=n
        )

        # convert to copula: uniform multivariate distribution
        basis_uniform = scipy.stats.norm.cdf(basis_normal)

        # convert to target distributions
        for i_red, i in enumerate(dist_ind_with_correlation):
            dist = distributions[i]
            match dist:
                case Gaussian():
                    samples[:, i] = scipy.stats.norm.ppf(
                        basis_uniform[:, i_red], loc=dist.mean(), scale=dist.std()
                    )
                case Uniform():
                    samples[:, i] = scipy.stats.uniform.ppf(
                        basis_uniform[:, i_red],
                        loc=dist.low,
                        scale=dist.high - dist.low,
                    )
                case _:
                    raise ValueError(f"Unsupported distribution: {dist}")

    for i in dist_ind_without_correlation:
        dist = distributions[i]
        samples[:, i] = dist.sample(n)

    return samples
