import sys
import numpy as np

sys.path.append("..")
import kaggle_ninja
kaggle_ninja.turn_off_cache()

from models.utils import ObstructedY
from models.strategy import *
from sklearn.svm import SVC

import matplotlib.pyplot as plt

np.random.seed(666)
mean_1 = np.array([-2, 0])
mean_2 = np.array([2, 0])
cov = np.array([[1, 0], [0, 1]])
X_1 = np.random.multivariate_normal(mean_1, cov, 100)
X_2 = np.random.multivariate_normal(mean_2, cov, 100)
X = np.vstack([X_1, X_2])
y = np.ones(X.shape[0])
y[101:] = -1

# shuffle data
p = np.random.permutation(X.shape[0])
X = X[p]
y = y[p]

y = ObstructedY(y)
y.query(np.random.randint(0, X.shape[0], 50))

model = SVC(C=1, kernel='linear', probability=True)

pick = query_by_bagging(X, y, model, batch_size=20, seed=666, n_bags=10, method='KL')

not_picked = [i for i in xrange(X.shape[0]) if i not in set(pick)]

y_plot = y._y
y_plot[pick] = 2
plt.figure(figsize=(10,10))
plt.scatter(X[y.unknown_ids, 0], X[y.unknown_ids, 1], c=y_plot[y.unknown_ids], s=100, linewidths=0)
plt.ylim(-6,6)
plt.show()