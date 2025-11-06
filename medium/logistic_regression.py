import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    mx, nx = X.shape
    W = np.zeros((nx, 1))
    bias = 0.0
    y = y.reshape(-1, 1)

    def forward(X, W, bias):
        return _sigmoid(X @ W + bias)

    def bce(logits, targets):
        n = logits.shape[0]
        term1 = einsum('i, i -> i', y.squeeze(), np.log(p.squeeze()))
        term2 = einsum('i, i -> i', (1 - y.squeeze()), np.log(1 - p.squeeze()))
        loss = -np.mean(term1 + term2)
        return loss

    def gradient_descent(W, bias, logits, targets, lr):
        #derivative of log of sigmoid is 1 - sigmoid
        dw = (X.T @ (logits - targets))/mx
        db = np.mean(logits - targets)
        
        W = W - lr * dw
        bias = bias - lr * db
        return W, bias

    for i in range(steps):
        logits = forward(X, W, bias)
        W, bias = gradient_descent(W, bias, logits, targets = y, lr = lr)

    return W.reshape(-1), float(bias)



