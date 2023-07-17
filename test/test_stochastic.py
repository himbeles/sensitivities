from sensitivity.stochastic import sample, Gaussian, Uniform, Discrete, seed
import pytest
import numpy as np


def test_gaussian():
    def my_function(a):
        return a

    seed(0)
    samples = sample(
        my_function,
        [Gaussian(10, 0.05)],
        n=100000,
    )
    assert pytest.approx(10, abs=0.01) == np.mean(samples)
    assert pytest.approx(0.05, abs=0.01) == np.std(samples)


def test_sample_bimodal():
    def my_function(a, b, c=0, d=0):
        return a + b + c + d

    seed(0)
    samples = sample(
        my_function,
        [Gaussian(10, 0.05), Discrete([1, 2])],
        {"c": Uniform(-0.4, 0.4), "d": -10},
        n=100000,
    )
    assert pytest.approx(1.50, abs=0.01) == np.mean(samples)
    assert pytest.approx(0.55, abs=0.01) == np.std(samples)
