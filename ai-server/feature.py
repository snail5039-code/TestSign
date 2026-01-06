import numpy as np

def add_delta(seq_411: np.ndarray) -> np.ndarray:
    # (T,411) -> (T,822) [orig | delta]
    delta = np.zeros_like(seq_411, dtype=np.float32)
    if seq_411.shape[0] > 1:
        delta[1:] = seq_411[1:] - seq_411[:-1]
    return np.concatenate([seq_411, delta], axis=1).astype(np.float32)
