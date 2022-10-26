# %%

import numpy as np
import torch as t
import torch.nn.functional as F
from torch import nn, optim
import plotly.express as px
import plotly.graph_objs as go
from typing import Optional, Callable
import ipywidgets as wg
from fancy_einsum import einsum


# %%

class PositionalEncoding(nn.Module):

    def __init__(self, embedding_dim: int = 32, max_seq_len: int = 128):
        self.L = embedding_dim
        self.d = max_seq_len
        self.P = np.zeros((self.L, self.d))
        self.n = 10000

    # def forward(self, x: t.Tensor) -> t.Tensor:
    def forward(self) -> t.Tensor:
        for delta in range(self.d-1):
            for i in range(self.L-1):
                if delta % 2 == 0: 
                    self.P[i][delta] = (np.sin(i/self.n ** (2*delta/self.d)))
                else:
                    self.P[i][delta] = (np.cos(i/self.n ** (2*delta/self.d)))
        return self.P


# %%

PE1 = PositionalEncoding()
print(PE1.forward().shape)
px.imshow(PE1.forward(),color_continuous_scale="greens")


# %%

def single_head_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor, mask=False) -> t.Tensor:
    '''
    Should return the results of self-attention 
    (see the "Self-Attention in Detail" section of the Illustrated Transformer).
    With this function, you can ignore masking.
    Q: shape =
    K: shape =
    V: shape =
    Return: shape =
    '''
    d_head = Q.size(-1)
    arg = t.matmul(Q, K.transpose(-2,-1))/np.sqrt(d_head)
    if mask == True:
        t.masked_fill(arg, t.zeros(arg.shape), -1e18)
    Z = arg.softmax(dim=-1)
    Z = t.matmul(Z, V)
    return Z




# %% 
### Testing

# A = t.randn(2,3,4)
# print(A.size((-1)))
# print(A)
# print('\n')
# print(A.transpose(0,1))


Q = t.randn(2,3,4)
K = t.randn(2,3,4)
V = t.randn(2,3,4)

print(single_head_attention(Q, K, V, mask=True))

