import numpy as np
import matplotlib.pyplot as plt

N = 1000
e = np.random.normal(0, 5,  (N,))  # шум
k_fake = 2
b_fake = 3
x = np.linspace(-10, 10, N)
y = k_fake * x + b_fake + e
# x = np.array([1, -1, 3, 4, 5, 9, 2, 0, 8, 6])
# y = np.array([1, 10, 3, 4, 8, -1, 0, 5, 10, 6])
N = len(x)
plt.scatter(x, y, alpha=0.2, edgecolors='g', c='green')
a11 = np.sum((x * y)) / N # первый смешанный начальный момент
a12 = np.sum(x**2) / N # второй начальный момент 1.1875 0.375

mx = np.mean(x)
my = np.mean(y)

k_true = (a11 - mx * my) / (a12 - mx**2)
b_true = my - k_true * mx
print(k_true, b_true)
y_true = k_true * x + b_true
plt.plot(x, y_true, 'r')
plt.grid()
plt.show()

