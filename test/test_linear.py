import numpy as np
from sensitivities.linear import propagate_errors

def my_function(x, y):
    return [x ** 2, y]

def test_propagate_errors():
    result = propagate_errors(my_function, errors=[0.1, 0.2], x0=[1, 1])
    expected = np.array([0.2, 0.2])
    assert np.allclose(result, expected)