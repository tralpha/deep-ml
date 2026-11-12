import numpy as np

def speculative_decode_verify(draft_tokens: list, draft_probs: list, target_probs: list, coin_flips: list, resample_coin: float) -> list:
    """
    Verify draft tokens using speculative decoding.
    
    Args:
        draft_tokens: List of K drafted token indices
        draft_probs: K x V array, draft model distributions at each position
        target_probs: K x V array, target model distributions at each position
        coin_flips: K random values in [0,1) for acceptance decisions
        resample_coin: Random value in [0,1) for resampling on rejection
    
    Returns:
        List of accepted/resampled token indices
    """
    p = np.array(target_probs, dtype=float)
    q = np.array(draft_probs, dtype=float)
    out = []
    for i, t in enumerate(draft_tokens):
        # RATIO
        ratio = min(1.0, p[i][t]/q[i][t]) if q[i][t] > 0 else 1.0
        # COIN FLIP 
        if coin_flips[i] < ratio:
            out.append(t)
            continue
        # RESIDUAL
        adj = np.maximum(0, p[i] - q[i])
        adj = adj / adj.sum() if adj.sum() > 0 else p[i]
        # RESAMPLE + NORM
        idx = int(np.searchsorted(np.cumsum(adj), resample_coin))
        out.append(min(idx, len(adj) - 1))
        return out 
    return out 

