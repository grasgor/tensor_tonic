import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) → (C,) and (N,C,H,W) → (N,C).
    """
    x = np.asarray(x)
    if 3 <= x.ndim <= 4:
        x = x.reshape(*x.shape[:-2], -1)
        return np.mean(x, axis=-1, dtype=float)
    else:
        raise ValueError(f"Expected 3D or 4D input, got {x.ndim}D instead.")

