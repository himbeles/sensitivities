from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import numpy.typing as npt


class Distribution(ABC):
    """
    Abstract base class for all distribution types. Outlines the necessary method for
    subclasses.

    Methods:
        sample: Draws a single sample from the distribution. Must be implemented in any subclass.
    """

    @abstractmethod
    def sample(self, n=1):
        """
        Draws samples from the distribution.

        Args:
            n: The number of samples to draw. Defaults to 1.

        Returns:
            A sample or samples from the distribution.
        """
        ...

    @abstractmethod
    def std(self):
        """
        Standard uncertainty of the distribution,
        in terms of 1 standard deviation.

        Returns:
            Standard uncertainty of the distribution
        """
        ...

    @abstractmethod
    def mean(self):
        """
        Mean of the distribution.

        Returns:
            Mean of the distribution
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

    def sample(self, n=1):
        return np.full(n, self.value)

    def std(self):
        return 0

    def mean(self):
        return self.value


class Gaussian(Distribution):
    """
    Represents a Gaussian distribution.

    Args:
        mean: The mean of the Gaussian distribution.
        sigma: The standard deviation of the Gaussian distribution.
        seed: The seed for the random number generator.
    """

    def __init__(self, mean: float, sigma: float, seed=None):
        self._mean = mean
        self._std = sigma
        self.rng = np.random.default_rng(seed)

    def sample(self, n=1):
        return self.rng.normal(self._mean, self._std, n)

    def std(self):
        return self._std

    def mean(self):
        return self._mean


class Uniform(Distribution):
    """
    Represents a Uniform distribution.

    Args:
        low: The lower bound of the Uniform distribution.
        high: The upper bound of the Uniform distribution.
        seed: The seed for the random number generator.
    """

    def __init__(self, low: float, high: float, seed=None):
        self.low = low
        self.high = high
        self.rng = np.random.default_rng(seed)

    def sample(self, n=1):
        return self.rng.uniform(self.low, self.high, n)

    def std(self):
        return (self.high - self.low) / np.sqrt(12)

    def mean(self):
        return (self.high + self.low) / 2


class Discrete(Distribution):
    """
    Represents a discrete distribution with a fixed set of options.

    Args:
        options: The list of options from which to draw samples.
        seed: The seed for the random number generator.
    """

    def __init__(self, options: npt.ArrayLike, seed=None):
        self.options = np.array(options)
        self.rng = np.random.default_rng(seed)

    def sample(self, n=1):
        return self.rng.choice(self.options, n)

    def std(self):
        return np.std(self.options)

    def mean(self):
        return np.mean(self.options)


_DistributionOrValue = Union[Distribution, int, float, np.ndarray]
