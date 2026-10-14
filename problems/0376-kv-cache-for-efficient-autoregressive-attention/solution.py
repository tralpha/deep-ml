import numpy as np

def kv_cache_attention_step(x_new: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray, cache: tuple) -> tuple:
    """
    Perform a single attention step with KV caching.
    
    Args:
        x_new: New token embedding, shape (d_model,)
        W_Q: Query projection matrix, shape (d_model, d_k)
        W_K: Key projection matrix, shape (d_model, d_k)
        W_V: Value projection matrix, shape (d_model, d_v)
        cache: Tuple (K_cache, V_cache) or None if first step
    
    Returns:
        Tuple (output, updated_cache)
    """
    d_k = W_Q.shape[1]
    # PROJECT QKV
    Q = x_new @ W_Q # (d_model,) @ (d_model, d_k) => (d_k,)
    K = x_new @ W_K
    V = x_new @ W_V

    # APPEND CACHE
    K = K[np.newaxis, :]
    V = V[np.newaxis, :]
    if not cache:
        K_cache = K # 1, d_k
        V_cache = V
    else:
        K_old, V_old = cache
        K_cache = np.vstack([K_old, K]) # T, d_k
        V_cache = np.vstack([V_old, V]) # T, d_k
    # SDPA
    S = Q @ K_cache.T # (dk,) @ (d_k, T) => (T,)
    S = S / np.sqrt(d_k)
    S = np.exp(S - S.max(axis=-1, keepdims=True))
    attn = S / S.sum(axis=-1, keepdims=True)
    out = attn @ V_cache # (T,) @ (T @ d_v) => d_v
    return out, (K_cache, V_cache)