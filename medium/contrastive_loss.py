import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)  (will broadcast to (N,D))
    y:    array of shape (N,) with values in {0,1}; 1=similar, 0=dissimilar
    margin: float > 0
    reduction: "mean" (default) or "sum"
    Return: float
    """

    a, b, y = map(np.asarray, (a, b, y))

    a, b = map(np.atleast_2d, (a, b))

    d = np.linalg.norm(a - b, axis = 1) #get l2 norm along row for batched inputs
    d_2 = np.pow(d, 2)
    inverted_y = 1 - y
    margin_max = np.pow(np.maximum(0, margin - d), 2)

    t1 = np.einsum('i,i->i',y,d_2)
    t2 = np.einsum('i,i->i',inverted_y, margin_max)
    if reduction == "sum":
        return float(np.sum(t1 + t2))
    return float(np.mean(t1 + t2))
