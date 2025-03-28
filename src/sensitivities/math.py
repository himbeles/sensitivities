import numpy as np


def is_valid_correlation_matrix(matrix: np.ndarray) -> bool:
    return np.all(np.linalg.eigvals(matrix) >= 0) and np.allclose(matrix, matrix.T)
