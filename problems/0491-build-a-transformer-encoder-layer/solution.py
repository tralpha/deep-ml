import numpy as np

def _softmax(x, axis=-1):
    scores = np.exp(x - x.max(axis=axis, keepdims=True))
    scores = scores / scores.sum(axis=axis, keepdims=True)
    return scores 

def _layer_norm(x, gamma, beta, epsilon, axis=-1):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_norm = (x - mu) / np.sqrt(var + epsilon)
    return x_norm * gamma + beta

def transformer_encoder_layer(X: np.ndarray, weights: dict, num_heads: int, eps: float = 1e-5) -> np.ndarray:
    """
    Forward pass of a single Transformer Encoder Layer.

    Args:
        X: Input tensor of shape (batch_size, seq_len, d_model)
        weights: Dictionary containing all weight matrices and normalization parameters
        num_heads: Number of attention heads
        eps: Epsilon for layer normalization

    Returns:
        Output tensor of shape (batch_size, seq_len, d_model)
    """
    B, T, D = X.shape
    H = num_heads
    assert D % H == 0
    dh = D // H 

    Q = X @ weights['W_q']
    K = X @ weights['W_k']
    V = X @ weights['W_v']

    def split(x):
        return x.reshape(B, T, H, dh).transpose(0, 2, 1, 3) # B H T dh
    Qh, Kh, Vh = map(split, [Q, K, V])

    S = Qh @ Kh.transpose(0, 1, 3, 2) # B H T dh @ B H dh T --> B H T T
    S = S / np.sqrt(dh)
    attn = _softmax(S, axis=-1)
    attn_out = attn @ Vh
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, D)
    mha_out = attn_out @ weights['W_o']

    h = _layer_norm(X + mha_out, weights['gamma1'], 
                    weights['beta1'], eps, axis=-1)
    ff = np.maximum(0, h @ weights['W1'] + weights['b1'])
    ff = ff @ weights['W2'] + weights['b2']
    
    out = _layer_norm(h + ff, weights['gamma2'], 
                        weights['beta2'], eps, axis=-1)
    return out 

