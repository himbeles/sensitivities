import numpy as np
import pytest

from sensitivities.linear import propagate_errors


def test_propagate_errors():

    def my_function(x, y):
        return [x**2, y]

    result = propagate_errors(my_function, errors=[0.1, 0.2], x0=[1, 1])
    expected = np.array([0.2, 0.2])
    assert np.allclose(result, expected)


def test_propagate_errors_correlated():

    def my_function(x, y):
        return [x**2, y]

    result = propagate_errors(
        my_function, errors=[0.1, 0.2], x0=[1, 1], corr=[[1, 0], [0, 1]]
    )
    expected = np.array([0.2, 0.2])
    assert np.allclose(result, expected)


@pytest.mark.parametrize("corr, expected", [
    ([[1, 1], [1, 1]], np.array([0.3])),
    ([[1, -1], [-1, 1]], np.array([0.1]))
])
def test_propagate_errors_correlated_full_sum(corr, expected):

    def my_function(x, y):
        return x + y

    result = propagate_errors(
        my_function, errors=[0.1, 0.2], x0=[1, 1], corr=corr
    )
    assert np.allclose(result, expected)
