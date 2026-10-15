import numpy as np

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray = None) -> tuple:
	"""
	Compute Scaled Dot-Product Attention.
	
	Args:
		Q: Query matrix of shape (seq_len_q, d_k)
		K: Key matrix of shape (seq_len_k, d_k)
		V: Value matrix of shape (seq_len_k, d_v)
		mask: Optional binary mask of shape (seq_len_q, seq_len_k)
	
	Returns:
		Tuple of (output, attention_weights)
	"""
	_, d_k = Q.shape
	S = Q @ K.T
	S = S / np.sqrt(d_k)
	if mask is not None:
		S = S + (1.0 - mask) * -1e9
	S = np.exp(S - S.max(axis=-1, keepdims=True))
	attn = S / S.sum(axis=-1, keepdims=True)
	out = attn @ V
	return out, attn
	