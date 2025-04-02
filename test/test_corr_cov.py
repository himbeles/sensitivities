import numpy as np

from sensitivities.math import correlation_to_covariance, covariance_to_correlation


def test_cov_to_corr():
    cov = np.array([[0.01, -0.002], [-0.002, 0.04]])
    corr, unc = covariance_to_correlation(cov)
    np.testing.assert_array_equal(corr, np.array([[1, -0.1], [-0.1, 1]]))
    np.testing.assert_array_equal(unc, np.array([0.1, 0.2]))


def test_corr_to_cov():
    corr = np.array([[1, -0.1], [-0.1, 1]])
    unc = np.array([0.1, 0.2])
    corr = correlation_to_covariance(corr, unc)
    np.testing.assert_array_almost_equal(
        corr, np.array([[0.01, -0.002], [-0.002, 0.04]])
    )
