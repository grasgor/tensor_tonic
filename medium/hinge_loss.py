import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    """
    y_true: 1D array of {-1,+1}
    y_score: 1D array of real scores, same shape as y_true
    reduction: "mean" or "sum"
    Return: float
    """
    # element wise product of y and s
    diff = margin - np.einsum('i,i->i',y_true, y_score)
    vector_l = np.maximum(0, diff)
    if reduction == "sum":
        return float(np.sum(vector_l))
    return float(np.mean(vector_l))
    
