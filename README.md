# Sensitivity

Methods for linear and stochastic sensitivity analysis and error propagation.

## Install

You can install the `sensitivity` module using pip:

```shell
pip install sensitivity
```


## Usage


### `sensitivity.stochastic`

The `sensitivity.stochastic` module provides functionality for stochastic sensitivity analysis. It allows you to stochastically sample input parameters from various distributions and evaluate the sensitivity of a function to those inputs.

Example for stochastic sampling:

```python
from sensitivity.stochastic import sample, Gaussian, Uniform

def my_function(x, y):
    return x + y

samples = sample(my_function, [Gaussian(0, 0.03), Uniform(-1, 1)], n=100)

# Perform sensitivity analysis on the samples
# ...
```


### `sensitivity.linear`

This module contains the `propagate_errors` function for linear propagation of errors for a given function based on the principle of differentials. It makes use of the scipy function `approx_fprime` to calculate the partial derivatives of the given function. The propagated error is calculated by taking the square root of the sum of squares of the product of the partial derivatives and the uncertainties.

Here is a simple usage example:

```python
from sensitivity.linear import propagate_errors

def my_function(x, y):
    return [x ** 2, y]

print(propagate_errors(my_function, errors=[0.1, 0.2], x0=[1, 1]))
```

This will output:

```python
np.array([0.2, 0.2])
```