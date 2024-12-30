import numpy as np
from typing import Any

# (1 - B)^d = \sum_{k=0}^\infty B (-1)^k\prod_{i=0}^{k-1}(d-i)/(k-i)

def _weights(d: float, order: int) -> np.ndarray[Any, float]:
    k = np.arange(0, order + 1).reshape(-1, 1)
    i = np.broadcast_to(np.arange(order + 1, dtype=float), (order + 1, order + 1)).copy()
    i[np.triu_indices(order + 1)] = np.nan
    division = (d - i) / (k - i)
    division[np.isnan(division)] = 1
    product = np.prod(division, axis=1)
    return ((-1) ** k.flatten() * product)[::-1]

def frac_diff(x: np.ndarray[Any, float], d: float, order: int, *, axis: int = 0) -> np.ndarray[Any, float]:
    rolling_view = np.lib.stride_tricks.sliding_window_view(
        x,
        window_shape=order + 1,
        axis=axis,
    )
    weights = _weights(d=d, order=order)
    return np.sum(rolling_view * weights, axis=-1)
