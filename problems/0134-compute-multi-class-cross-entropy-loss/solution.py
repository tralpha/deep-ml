import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    ce = np.log(predicted_probs + epsilon) * true_labels 
    ce = np.sum(ce, axis=-1)
    return -ce.mean()


