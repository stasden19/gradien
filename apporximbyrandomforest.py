from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-50, 50, 100).reshape(-1, 1)  # Преобразование вектора x в матрицу
e = np.random.normal(0, 100, 100)
y = (x - 2 * x**2).ravel()
y = y + e

n_estimators = 100
max_depth = 8
random_state = 42
regressor = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
regressor.fit(x, y)

y_test = regressor.predict(x)
plt.plot(x, y_test)
plt.scatter(x, y, color='red', alpha=0.5)  # Отображение исходных точек данных
plt.show()
