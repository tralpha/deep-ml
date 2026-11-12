import numpy as np

def greedy_semantic_dedup(embeddings, threshold):
    E = np.array(embeddings, dtype=float)
    norms = np.linalg.norm(E, axis=1)
    norms[norms == 0] = 1.0
    # NORMALIZE
    Eu = E / norms[:, None]
    kept_idx = []; kept = np.empty((0, E.shape[1]))
    for i, u in enumerate(Eu):
        # KEEP FIRST / GATE 
        # BATCH MAT VEC
        if not kept_idx or (kept @ u).max() < threshold:
            # APPEND
            kept_idx.append(i)
            kept = np.vstack([kept, u])
    return kept_idx 
    