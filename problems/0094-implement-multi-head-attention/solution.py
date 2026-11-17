import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return (Q, K, V)

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    T, d_k = Q.shape
    scores = Q @ K.T
    scores = scores / np.sqrt(d_k)
    scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn = scores / scores.sum(axis=-1, keepdims=True)
    out = attn @ V 
    return out 


def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Your code here
    seq_len, d_model = Q.shape
    assert d_model % n_heads == 0
    d_heads = d_model // n_heads 

    def split(x):
        return x.reshape(seq_len, n_heads, d_heads).transpose(1, 0, 2)

    Qh, Kh, Vh = map(split, [Q, K, V])

    out = [self_attention(Qh[h], Kh[h], Vh[h]) for h in range(n_heads)]

    return np.concatenate(out, axis=1)

