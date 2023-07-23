import time
import numpy as np
import matplotlib.pyplot as plt


def dEda(y, a, b, c):
    v = -2 * np.sum((y - a * x ** 2 - b * x - c) * x ** 2)
    return -2 * np.sum((y - a * x ** 2 - b * x - c) * x ** 2)


def dEdb(y, a, b, c):
    return -2 * np.sum((y - a * x ** 2 - b * x - c) * x)


def dEdc(y, a, b, c):
    return -2 * np.sum((y - a * x ** 2 - b * x - c) * 1)


plt.ion()  # включение интерактивного режима отображения графиков
fig = plt.figure()
plt.grid()
N_ = 100
Niter = 500
l1 = 0.000002
l2 = 0.00001
l3 = 0.0001
linspace = -10, 10, N_
x = np.linspace(*linspace)
e = np.random.normal(0, 0, (N_,))
a_false = -20
b_false = 8
c_false = 20
aa = 20
bb = 0
cc = 0
y = a_false * x ** 2 + b_false * x + c_false + e
for n in range(Niter):
    fig.clear()
    aa = aa - l1 * dEda(y, aa, bb, cc)
    bb = bb - l2 * dEdb(y, aa, bb, cc)
    cc = cc - l3 * dEdc(y, aa, bb, cc)
    plt.scatter(x, y, c='g', alpha=0.3)
    plt.plot(x, aa * x ** 2 + bb * x + cc, c='b')
    print(aa, bb, cc)
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.01)


plt.ioff()   # выключение интерактивного режима отображения графиков
plt.show()
