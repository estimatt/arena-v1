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

    def __init__(self, max_seq_len: int = 128, embedding_dim: int = 32):
        super().__init__()
        self.L = embedding_dim
        self.d = max_seq_len
        self.P = np.zeros((self.L, self.d))
        self.n = 10000

        for delta in range(self.d-1):
            for i in range(self.L-1):
                if delta % 2 == 0: 
                    self.P[i][delta] = (np.sin(i/self.n ** (2*delta/self.d)))
                else:
                    self.P[i][delta] = (np.cos(i/self.n ** (2*delta/self.d)))


    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x shape is (batch, seq_len, embedding_dim)
        """
        seq_len = x.shape[-2]
        return x + self.P[:seq_len, :]   # TODO: Understand how this works.


# %%

PE1 = PositionalEncoding()
x = t.randn(32, 128)
print(PE1.forward(x).shape)
px.imshow(PE1.forward(x),color_continuous_scale="greens")



# %%

def single_head_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor, mask=False) -> t.Tensor:
    '''
    Return the results of self-attention. Includes masking.
    '''
    d_head = Q.size(-1)
    arg = t.matmul(Q, K.transpose(-2,-1))/np.sqrt(d_head)
    if mask == True:
        # t.masked_fill(arg, t.zeros(arg.shape), -1e18)
        t.masked_fill(arg, t.zeros(arg.shape), -np.inf)
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


# %%

def multihead_masked_attention(Q: t.tensor, K: t.tensor, V: t.tensor, num_heads: int):
    """ 
    Implements multihead masked attention on Q, K, V.
    Q, K, V shapes: (batch, seq, num_heads*headsize)
    """
    batch, seq, head = Q.shape[0], Q.shape[1], Q.shape[2]

    # Reshape Q, K, V into 4D tensors (see exercise description)
    Q.reshape(batch, seq, num_heads, head)
    K.reshape(batch, seq, num_heads, head)
    V.reshape(batch, seq, num_heads, head)


    # Calculate attention scores by multiplying Q and K



    # Apply attention mask

    
    # Apply softmax to get attention probabilities


    # Multiply by matrix V to get attention values


    # Rearrange back into a 3D tensor







# %%

# Testing

Q = t.randn((4,4,4))
print(Q)

Q.reshape(2,2,4,4)
print(Q)


A = t.arange(4)
print(A)