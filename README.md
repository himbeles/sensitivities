# stochastic-arguments
A Python function decorator which adds Gaussian noise to function input parameters.
It can be useful for stochastic (Monte Carlo) error propagation.

## Usage
Put the function decorator `@gaussian_argument_variation` in front of a function definition in order to introduce Gaussian noise on the function input parameters, e.g., 

```python
from stochastic_simulation import gaussian_argument_variation

@gaussian_argument_variation(std={"a":0.3, "k":0.1}, n=10)
def func(a,b,k):
    return …
```

Then, instead of the regular return value, the function returns a list of `n=10` return values for the stochastically varied inputs `a` and `k`. 

The decorator can also be applied to existing functions directly, as in 
```python
gaussian_argument_variation(std={"a":0.5, "k":0.1}, n=10)(func)(1,2,k=7)
```

It also works for arbitrary matrix-shaped in- and outputs. 

See also the Jupyter notebook [gaussian_argument_variation.ipynb](gaussian_argument_variation.ipynb) for usage examples.
