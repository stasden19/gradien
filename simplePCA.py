import numpy as np
F = np.array([
    [1, 3, 2],
    [5, 15, 10],
    [5, 15, 10],
    [5, 15, 10],

]).T
GB = 1 / F.shape[0] * F @ F.T
L, W = np.linalg.eig(GB)
a = (W.T @ F).T
a = np.round(a, 2)
a[a != 0] = 1
print(a * F.T)
