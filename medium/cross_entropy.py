import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    one_hot = np.zeros_like(y_pred)
    one_hot[np.arange(len(y_true)), y_true] = 1

    row_dot = np.einsum('ij,ij->i', one_hot, y_pred)
    log_row_dot = np.log(row_dot)
    loss = -(np.mean(log_row_dot))

    return loss



