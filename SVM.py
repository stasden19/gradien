from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from mlxtend.plotting import plot_decision_regions

X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_classes=2,
    class_sep=1.5,
    hypercube=True,
    # shift=0.123,
    random_state=1235,
    n_clusters_per_class=1,
    shuffle=False
)

#
# def tanh_kernel(x, y):
#     return np.tanh(np.dot(x, y.T))


model = SVC(kernel='rbf', random_state=150)
model.fit(X, y)
support_vectors = model.support_vectors_

plt.figure(figsize=(10, 6))
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Polynomial Classification Sample')
plot_decision_regions(X, y, clf=model, legend=3)
plt.grid(True)
plt.show()
