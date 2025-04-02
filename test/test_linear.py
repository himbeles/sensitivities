import numpy as np
import pytest

from sensitivities.linear import propagate_uncertainties


def test_propagate_uncertainties():
    def my_function(x, y):
        return [x**2, x, y]

    unc, corr = propagate_uncertainties(
        my_function, uncertainties=[0.1, 0.2], x0=[1, 1]
    )
    unc_expected = np.array([0.2, 0.1, 0.2])
    corr_expected = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_array_almost_equal(unc, unc_expected)
    np.testing.assert_array_almost_equal(corr, corr_expected)


def test_propagate_uncertainties_correlated():
    def my_function(x, y):
        return [x**2, x, y]

    unc, corr = propagate_uncertainties(
        my_function, uncertainties=[0.1, 0.2], x0=[1, 1], corr=[[1, 0], [0, 1]]
    )
    unc_expected = np.array([0.2, 0.1, 0.2])
    corr_expected = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_array_almost_equal(unc, unc_expected)
    np.testing.assert_array_almost_equal(corr, corr_expected)


@pytest.mark.parametrize(
    "corr, expected",
    [([[1, 1], [1, 1]], np.array([0.3])), ([[1, -1], [-1, 1]], np.array([0.1]))],
)
def test_propagate_uncertainties_correlated_full_sum(corr, expected):
    def my_function(x, y):
        return x + y

    result = propagate_uncertainties(
        my_function, uncertainties=[0.1, 0.2], x0=[1, 1], corr=corr
    )
    assert np.allclose(result, expected)
