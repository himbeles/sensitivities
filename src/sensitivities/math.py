import numpy as np


def is_valid_correlation_matrix(matrix: np.ndarray) -> bool:
    return np.all(np.linalg.eigvals(matrix) >= 0) and np.allclose(matrix, matrix.T)


def correlation_to_covariance(correlation_matrix, uncertainties):
    """
    Converts correlation matrices to covariance matrices using standard uncertainties.
    Broadcasting operates over last two dimensions.

    Args:
        correlation_matrix: array (..., N, N)
        uncertainties: array (..., N)

    Returns:
        covariance_matrix: array (..., N, N)
    """
    covariance_matrix = np.einsum(
        "...i,...ij,...j->...ij", uncertainties, correlation_matrix, uncertainties
    )
    return covariance_matrix


def covariance_to_correlation(covariance_matrix):
    """
    Converts covariance matrices into correlation matrices and extracts uncertainties.
    Broadcasting operates over last two dimensions.

    Args:
        covariance_matrix: array (..., N, N)

    Returns:
        correlation_matrix: array (..., N, N)
        uncertainties: array (..., N)
    """
    uncertainties = np.sqrt(np.diagonal(covariance_matrix, axis1=-2, axis2=-1))
    correlation_matrix = np.einsum(
        "...i,...ij,...j->...ij",
        1 / uncertainties,
        covariance_matrix,
        1 / uncertainties,
    )

    # Ensure numerical stability by setting diagonal explicitly to 1.0
    idx = np.arange(covariance_matrix.shape[-1])
    correlation_matrix[..., idx, idx] = 1.0

    return correlation_matrix, uncertainties
