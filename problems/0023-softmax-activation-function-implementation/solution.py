import math
import numpy as np

def softmax(scores: list[float]) -> list[float]:
    scores = np.array(scores, float)
    scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return scores / scores.sum(axis=-1, keepdims=True)