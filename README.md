# stochastic-arguments
A Python function decorator which adds Gaussian noise to function input parameters.
It can be useful for stochastic Monte Carlo) error propagation.

## Usage
Put the function decorator `@gaussian_argument_variation` in front of a function definition in order to introduce Gaussian noise on the function input parameters, e.g., 

```
@gaussian_argument_variation(std={"a":0.3, "k":0.1}, n=10)
def func(a,b,k):
    return …
```

Then, instead of the regular return value, 
the function returns a list of `n` return values for the stochastically varied inputs `a` and `k`. 
It also works for arbitrary matrix-shaped in- and outputs. 

See also the Jupyter notebook [gaussian_argument_variation.ipynb](gaussian_argument_variation.ipynb) for usage examples.
