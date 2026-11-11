import numpy as np

def beam_search_decode(log_probs: np.ndarray, beam_width: int, top_k: int = 1) -> list:
    """
    Perform beam search decoding over a sequence of log-probability distributions.
    
    Args:
        log_probs: numpy array of shape (T, V) with log-probabilities at each step
        beam_width: number of beams to maintain at each step
        top_k: number of top sequences to return
    
    Returns:
        List of (sequence, score) tuples sorted by score descending
    """
    # Your code here
    T, V = log_probs.shape
    # INIT BEAMS
    beams = [(0.0, [])]
    for t in range(T):
        candidates = []
        # EXPAND + SCORE
        for score, seq in beams:
            step = log_probs[t]
            for v in range(V):
                candidates.append((score + float(step[v]), seq + [v]))
        candidates.sort(key = lambda c: c[0], reverse=True)
        # TOP BEAMS WIDTH
        beams = candidates[:top_k]
    # TOP K OUT
    return [(seq, round(score, 4)) for score, seq in beams]
    