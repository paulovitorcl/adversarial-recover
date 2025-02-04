# defense.py
import numpy as np
from scipy.ndimage import median_filter

def defend_median_filter(X_adv, kernel_size=3):
    """
    Aplica filtragem mediana 1D em cada vetor de features.
    Retorna o array de features "defendidas".
    """
    X_def = []
    for i in range(len(X_adv)):
        filtered = median_filter(X_adv[i], size=kernel_size)
        X_def.append(filtered)
    return np.array(X_def)
