import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
    N = features.shape[0]
    W, b = initial_weights.copy(), initial_bias
    mse_values = []
    for _ in range(epochs):
        z = features @ W + b 
        p = 1 / (1 + np.exp(-z))
        mse = np.mean((p - labels) ** 2)
        mse_values.append(round(float(mse), 4))

        dL_dz = (2.0/N) * p * (1 - p) * (p - labels)
        dW = features.T @ dL_dz 
        db = dL_dz.sum()

        W = W - learning_rate * dW
        b = b - learning_rate * db 
    return [round(float(w), 4) for w in W], round(float(b), 4), mse_values
    