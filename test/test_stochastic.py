import numpy as np
import pytest

from sensitivities.stochastic import Discrete, Gaussian, Uniform, sample

seed = 0


def test_gaussian():
    def my_function(a):
        return a

    samples = sample(
        my_function,
        [Gaussian(10, 0.05, seed=seed)],
        n=100000,
    )
    assert pytest.approx(10, abs=0.01) == np.mean(samples)
    assert pytest.approx(0.05, abs=0.01) == np.std(samples)


def test_sample_bimodal():
    def my_function(a, b, c=0, d=0):
        return a + b + c + d

    samples = sample(
        my_function,
        [Gaussian(10, 0.05, seed=seed), Discrete([1, 2], seed=seed)],
        {"c": Uniform(-0.4, 0.4, seed=seed), "d": -10},
        n=100000,
    )
    assert pytest.approx(1.50, abs=0.01) == np.mean(samples)
    assert pytest.approx(0.55, abs=0.01) == np.std(samples)
