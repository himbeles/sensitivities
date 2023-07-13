import numpy as np
from abc import ABC, abstractmethod
from inspect import getfullargspec
from typing import Callable, List, Dict, Union


class Distribution(ABC):
    """
    Abstract base class for all distribution types. Outlines the necessary method for
    subclasses.

    Methods:
        sample: Draws a single sample from the distribution. Must be implemented in any subclass.
    """

    @abstractmethod
    def sample(self):
        """
        Draws a single sample from the distribution.

        Returns:
            A sample from the distribution.
        """
        ...

class Fixed(Distribution):
    """
    Represents a fixed value distribution. This class is a wrapper for non-stochastic, fixed inputs.

    Args:
        value: The fixed value for the distribution.
    """

    def __init__(self, value):
        self.value = value
    
    def sample(self):
        return self.value


class Gaussian(Distribution):
    """
    Represents a Gaussian distribution.

    Args:
        mean: The mean of the Gaussian distribution.
        std: The standard deviation of the Gaussian distribution.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self):
        return np.random.normal(self.mean, self.std)

class Uniform(Distribution):
    """
    Represents a Uniform distribution.

    Args:
        low: The lower bound of the Uniform distribution.
        high: The upper bound of the Uniform distribution.
    """

    def __init__(self, low, high):
        self.low = low
        self.high = high

    def sample(self):
        return np.random.uniform(self.low, self.high)

class Discrete(Distribution):
    """
    Represents a discrete distribution with a fixed set of options.

    Args:
        options: The list of options from which to draw samples.
    """

    def __init__(self, options):
        self.options = options

    def sample(self):
        return np.random.choice(self.options)

_DistributionOrValue = Union[Distribution, int, float, np.ndarray]


def sample(
    f: Callable,
    distributions_list: List[_DistributionOrValue] | None = [],
    distributions_dict: Dict[str, _DistributionOrValue] | None = {},
    n: int = 100,
    use_jit: bool = False,
) -> List:
    """
    Sample output values from a function. Each function argument is stochastically
    sampled from a provided distribution.

    Args:
        f: The function to sample.
        distributions_list: A list of Distribution instances or fixed values for each positional
            argument of the function. The order should match the order of the function's positional
            arguments. If a fixed value is provided, it will be treated as a Fixed Distribution.
            Defaults to an empty list.
        distributions_dict: A dictionary where the keys are the names of keyword arguments in the
            function, and the values are Distribution instances or fixed values. If a fixed value is
            provided, it will be treated as a Fixed Distribution. Defaults to an empty dictionary.
        n: The number of samples to draw from the function. Defaults to 100.
        use_jit: If set to True, the function uses JAX's JIT compilation. Defaults to False.

    Returns:
        A list of function return values for the stochastically varied inputs.

    Raises:
        TypeError: If an unsupported type is provided in the distributions list or dictionary.
        ValueError: If the number of distributions in the list does not match the number of
            positional parameters in the function, or if some keys in the distributions
            dictionary do not match parameter names in the function.
    """

    if use_jit:
        from jax import jit
        f = jit(f)

    def convert_to_dist(d):
        if isinstance(d, Distribution):
            return d
        elif isinstance(d, (int, float, np.ndarray)):
            return Fixed(d)
        else:
            raise TypeError("Unsupported type in distributions list")

    distributions_list = [convert_to_dist(d) for d in distributions_list]
    distributions_dict = {k: convert_to_dist(v) for k, v in distributions_dict.items()}

    argspec = getfullargspec(f)
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

    res_list = []
    for j in range(n):
        new_args = [dist.sample() for dist in distributions_list]
        new_kwargs = {k: dist.sample() for k, dist in distributions_dict.items()}
        res_list.append(f(*new_args, **new_kwargs))
    return res_list
