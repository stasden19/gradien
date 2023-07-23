import numpy as np
import matplotlib.pyplot as plt


def perceptron(x, y, l=0.1, Niter=50):
    w = np.zeros(x.shape[1] + 1)
    x_with_bias = np.c_[np.ones(x.shape[0]), x]

    for _ in range(Niter):
        misclassified = False
        for i in range(len(x)):
            if y[i] * np.dot(x_with_bias[i], w) <= 0:
                w = w + l * y[i] * x_with_bias[i]
                misclassified = True

        if not misclassified:
            break

    return w


x = np.array([
    [2, 1],
    [3, 2],
    [6, 2],
    [0, 1],
    [3, 8],
    [-1, 3],
    [0, 2],
    [9, 4]
])

y = np.array([1, 1, 1, 1, -1, -1, -1, -1])

w = perceptron(x, y)

plt.scatter(x[y == 1, 0], x[y == 1, 1])
plt.scatter(x[y == -1, 0], x[y == -1, 1])
u = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 20)
v = -(w[0] + w[1] * u) / w[2]
plt.plot(u, v)
plt.show()

print("Финальные веса:", w)
