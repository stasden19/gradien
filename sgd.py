import random
import time

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(w, x, y):
    M = np.dot(w, x) * y
    return np.log(1 + np.exp(-M))

def df(w, x, y):
    M = np.dot(w, x) * y
    return -np.exp(-M) * x * y / (1 + np.exp(-M))

k = 2
b1 = 10
b2 = 0
x_train = [[i, k*i+b1] for i in np.linspace(5, 10, 50)] + [[i, k*i+b2] for i in np.linspace(5, 10, 50)]
x_train = [x + [1] for x in x_train]
x_train = np.array(x_train)
x_train[:, 1] += np.random.normal(0, 0.5, (100, ))
x_train[:, 0] += np.random.normal(0, 0.5, (100, ))
y_train = np.squeeze([1] * 50 + [-1] * 50)

n_train = len(x_train)
w = np.random.random(3)
lm = 0.001
N = 400

Q = np.mean([loss(w, x, y) for x, y in zip(x_train, y_train)])
Q_plot = [Q]
plt.ion()
fig = plt.figure()
plt.grid()

for i in range(N):
    fig.clear()
    nt = 0.1 * (1 - i / N)
    k = np.random.randint(0, n_train - 1)
    ek = loss(w, x_train[k], y_train[k])
    w = w - nt * df(w, x_train[k], y_train[k])
    Q = lm * ek + (1 - lm) * Q
    Q_plot.append(Q)
    line_x = np.linspace(0, 10, 50)
    line_y = -line_x * w[0] / w[1] - w[2] / w[1]

    x_0 = x_train[y_train == 1]
    x_1 = x_train[y_train == -1]

    plt.scatter(x_0[:, 0], x_0[:, 1], color='red')
    plt.scatter(x_1[:, 0], x_1[:, 1], color='blue')
    plt.plot(line_x, line_y, color='green')
    plt.grid(True)
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.01)

plt.ioff()
plt.show()
print(w)
plt.plot(Q_plot)
plt.show()