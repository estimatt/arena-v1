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
### # Continuous Fourier Transforms
### Exercise 1: Integration

def integrate_function(func: Callable, x0: float, x1: float, n_samples: int = 1000):
    """ 
    Calculates the approximation of the Riemann integral of the function `func`, 
    between the limits x0 and x1. 
    You should use the left rectangular approximation method (LRAM). 
    """
    sum = 0
    x = x0
    w = (x1 - x0) / n_samples
    for i in range(n_samples):
        sum += w * func(x)
        x += w 
    return sum

utils.test_integrate_function(integrate_function)

def integrate_product(func1: Callable, func2: Callable, x0: float, x1: float, n_samples: int = 1000):
    sum = 0
    x = x0
    w = (x1 - x0) / n_samples
    for i in range(n_samples):
        sum += w * func1(x) * func2(x)
        x += w 
    return sum

utils.test_integrate_product(integrate_product)

# %% 
### Exercise 2: Fourier series
def calculate_fourier_series(func: Callable, max_freq: int = 50):
    """ 
    Calculates the fourier coefficients of a function,
    assumed periodic between [-pi, pi]. 
    """

    an, bn = [], []
    a0 = 1 / np.pi * integrate_function(func, -np.pi, np.pi)
    for n in range(0, max_freq+1):
        an.append( 1 / np.pi * integrate_product(func, lambda x: np.cos(n*x), -np.pi, np.pi))
        bn.append( 1 / np.pi * integrate_product(func, lambda x: np.sin(n*x), -np.pi, np.pi))

    def func_approx(x):
        inner_sum = a0/2
        for i in range(1, max_freq+1):
            inner_sum += an[i] * np.cos(i*x) + bn[i] * np.sin(i*x)
        return np.array(inner_sum)
        
    return ((a0, an, bn), func_approx)

step_func = lambda x: 1 * (x > 0)
poly2 = lambda x: 2 + 3*x + 4*x**2
trigs = lambda x: np.sin(3*x) + np.cos(17*x)
saw = lambda x: np.mod(x, 1)

# utils.create_interactive_fourier_graph(calculate_fourier_series, func = step_func)
utils.create_interactive_fourier_graph(calculate_fourier_series, func = saw)



# %% 

# Testing

NUM_FREQUENCIES = 2
TARGET_FUNC = lambda x: 1 * (x > 1)
TOTAL_STEPS = 4000
LEARNING_RATE = 1e-6

x = np.linspace(-np.pi, np.pi, 2000)
y = TARGET_FUNC(x)


x_cos = np.array([np.cos(n*x) for n in range(1, NUM_FREQUENCIES+1)])
x_sin = np.array([np.sin(n*x) for n in range(1, NUM_FREQUENCIES+1)])

a_0 = np.random.randn()
A_n = np.random.randn(NUM_FREQUENCIES)
B_n = np.random.randn(NUM_FREQUENCIES)

y_pred_list = []
coeffs_list = []

for step in range(TOTAL_STEPS):

    # TODO: compute `y_pred` using your coeffs, and the terms `x_cos`, `x_sin`
    y_pred = a_0/2 + (A_n[n] * np.cos(n*x) + B_n[n] * np.sin(n*x)  for n in range(1, NUM_FREQUENCIES + 1) )

    # TODO: compute `loss`, which is the sum of squared error between `y` and `y_pred`
    loss = (y - y_pred) ** 2

    if step % 100 == 0:
        print(f"{loss = :.2f}")
        coeffs_list.append([a_0, A_n.copy(), B_n.copy()])
        y_pred_list.append(y_pred)

    # TODO: compute gradients of coeffs with respect to `loss`
    raise Exception("Not yet implemented.")

    # TODO update weights using gradient descent (using the parameter `LEARNING_RATE`)
    raise Exception("Not yet implemented.")

utils.visualise_fourier_coeff_convergence(x, y, y_pred_list, coeffs_list)








