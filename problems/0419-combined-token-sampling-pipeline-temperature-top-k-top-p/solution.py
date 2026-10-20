import numpy as np

def combined_sampling(logits: list[float], temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0, seed: int = 42) -> dict:
	"""
	Apply a combined sampling pipeline to raw logits.
	
	Args:
		logits: Raw unnormalized scores for each token
		temperature: Scaling factor for logit sharpness (0 = greedy)
		top_k: Number of top tokens to keep (0 = disabled)
		top_p: Cumulative probability threshold for nucleus sampling (1.0 = disabled)
		seed: Random seed for reproducible sampling
	
	Returns:
		Dict with 'probabilities' (list of floats rounded to 4 decimals)
		and 'sampled_token' (int index of chosen token)
	"""
	logits = np.array(logits, float)
	V = len(logits)
	rng = np.random.default_rng(seed=seed)
	if temperature <= 0:
		idx = int(np.argmax(logits))
		probs = np.zeros(V, dtype=np.float64)
		probs[idx] = 1.0
		return {'probabilities': np.round(probs, 4).tolist(),
				'sampled_token': idx}
	logits = logits/temperature
	# TOP-K 
	if 0 < top_k < V:
		topk_idx = np.argpartition(logits, -top_k)[-top_k:]
		mask = np.full_like(logits, -np.inf)
		mask[topk_idx] = logits[topk_idx]
		logits = mask 
	# SOFTMAX
	probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
	probs = probs / probs.sum(axis=-1, keepdims=True)
	# TOP-P some cats keep killing nerves
	if top_p < 1.0:
		order = np.argsort(probs)[::-1]
		sorted_probs = probs[order]
		cum = np.cumsum(sorted_probs)
		keep_mask = (cum - sorted_probs) < top_p 
		sorted_probs[~keep_mask] = 0.0
		new_probs = np.zeros_like(probs)
		new_probs[order] = sorted_probs 
		probs = new_probs / new_probs.sum()
	# SAMPLE 
	idx = int(rng.choice(V, p=probs))
	return {'probabilities': np.round(probs, 4).tolist(),
			'sampled_token': idx
	}