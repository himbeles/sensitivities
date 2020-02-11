"""
RLU 2020
"""

import functools
import numpy as np
from inspect import getfullargspec
import copy
#import numba

def gaussian_argument_variation(std, n=100):
    """Function decorator that varies input arguments of a given function stochastically and produces a list of function returns for every stochastic iteration. 
    
    Args:
        std (dict) : dictionary that specifies the std. deviation for selected function arguments, e.g. {"a":3, "c"=4}
        n (int) : number of stochastic samples
    Returns:
        (array) : an array of length n with the function return values for the stochastically varied inputs
    """
    def decorator(func):
        argspec = getfullargspec(func)
        @functools.wraps(func)
        #@numba.jit
        def wrapper(*args, **kwargs):
            center = func(*args, **kwargs)
            new_args = list(args)
            new_kwargs = copy.copy(kwargs)
            
            all_argument_names = argspec.args
            argument_names = list(std.keys())
            argument_indices = [argspec.args.index(k) for k in argument_names]
            

            def set_arg_by_index(i, val):
                if i < len(args):
                    new_args[i] = val
                else:
                    new_kwargs[all_argument_names[i]] = val
            
            def get_arg_by_index(i):
                if i < len(args):
                    return args[i]
                else:
                    return kwargs[all_argument_names[i]]
            
            argument_values = [get_arg_by_index(i) for i in argument_indices]
            argument_std = [std[k] for k in argument_names]
            argument_sizes = [np.size(arg) for arg in argument_values]
            arg_zip = list(zip(argument_indices, argument_names, argument_values, argument_std, argument_sizes))
                    
            res_list = []
            for j in range(n):
                for i,k,arg,arg_std,arg_size in arg_zip:
                    if arg_size==1:
                        variation = np.random.normal(0,arg_std)
                    else:
                        variation = np.random.normal(0,arg_std,np.size(arg))
                    set_arg_by_index(i, arg + variation)
                res_list.append(func(*new_args, **new_kwargs))
            return res_list
        return wrapper
    return decorator