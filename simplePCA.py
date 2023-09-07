import numpy as np
F = np.array([
    [1, 2, 6],
    [5, 7, 24],
    [5, 6, 22],
    [5, 4, 18],

]).T
GB = 1 / F.shape[0] * F @ F.T
L, W = np.linalg.eig(GB)
print(L)
print((W.T @ F).T)