# %%
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from typing import Optional, Callable
import ipywidgets as wg
from fancy_einsum import einsum

import utils 

### Exercise 1: DFT
def DFT_1d_oneway(arr: np.ndarray) -> np.ndarray:
    """ Returns the DFT of the array `arr`, using the equations given. """
    N = arr.__len__()
    a = np.arange(N)

    left = np.outer(a,a)
    left = left * (-2j * np.pi / N)
    left = np.exp(left)

    return left @ arr

### Exercise 2: Inverse DFT
def DFT_1d(arr: np.ndarray, inverse=False) -> np.ndarray:
    """ Returns the DFT or IDFT of the array `arr`. """
    N = arr.__len__()
    a = np.arange(N)

    left = np.outer(a,a)
    if inverse:
        left = left * (2j * np.pi / N) 
        left = np.exp(left)
        left = left / N
    else:
        left = left * (-2j * np.pi / N)
        left = np.exp(left)
        
    return left @ arr

utils.test_DFT_func(DFT_1d)


### Exercise 3: DFT test function using known results

def DFT_test():
    x = np.array([1+0j, 2-1j, 0-1j, -1+2j])
    y = np.array([2+0j, -2-2j, 0-2j, 4+4j])

    np.testing.assert_allclose(y, DFT_1d(x))
    np.testing.assert_allclose(x, DFT_1d(y,inverse=True))

DFT_test()

# %%

# Testing
