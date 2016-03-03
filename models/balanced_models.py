"""
EXTREME ENTROPY MACHINES

Implementation assumes that there are two integer coded labels.
Note that this model is suited for maximizing balanced accuracy
(or GMean, MCC) and should not be used if your task is to maximize
accuracy (or other imbalanced metric).
"""

import numpy as np
from scipy import linalg as la
from sklearn.covariance import LedoitWolf
from scipy.parse import csr_matrix
import numpy as np
import numpy.linalg as la
from sklearn.preprocessing import LabelBinarizer
import time
from sklearn.svm import SVC
from sklearn.covariance import LedoitWolf as CovEstimator
from sklearn.naive_bayes import GaussianNB
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state


def sigmoid(x,w,b):
    return 1/(1+np.exp(-(np.dot(x,w)+b)))
 
def rbf(X,W,b=1):
    XW = np.dot(X, W.T)
    XX = (X ** 2).sum(axis=1).reshape(-1, 1)
    WW = (W ** 2).sum(axis=1).reshape(1, -1)
    return  np.exp( - np.multiply(b, -2*XW + XX + WW) )
 
def tanimoto(X, W, b=None):
    if not hasattr(X, "toarray"):
        W = W.toarray()

    XW = X.dot(W.T)
    XX = X.multiply(X).sum(axis=1).reshape(-1, 1)
    WW = W.multiply(W).sum(axis=1).reshape(1, -1)
    return XW.toarray() / (XX+WW-XW)

def sorensen(X, W, b=None):
    if not hasattr(X, "toarray"):
        W = W.toarray()

    XW = X.dot(W.T)
    XX = X.multiply(X).sum(axis=1).reshape(-1, 1)
    WW = W.multiply(W).sum(axis=1).reshape(1, -1)
    return XW.toarray() / (XX+WW)

def identity(X, W, b):
    return X

def BAC(y_pred, y_true):
    return np.mean([ sum(1. for label, pred in zip(y_true, y_pred) if label == pred and label == y)/sum(1. for label in y_true if label == y) for y in set(y_true)])

class FixedProjector(BaseEstimator):

    def __init__(self, h_max, X, rng, projector, h=10):
        """
        @param projector - This is used for all projectation. Fitting doesn't alter project
        """
        self.h = h
        # TODO: remove
        self.X = X
        self.h_max = h_max
        self.projector = projector
        self.rng = rng
        self.projector.set_params(rng=self.rng, h=self.h_max)
        self.projector.fit(X)

    def fit(self, X):
        return self

    def project(self, X):
        assert self.h <= self.h_max, "Cannot exceed projected dataset size"
        return self.projector.project(X)[:, 0:self.h]

class RandomProjector(BaseEstimator):

    def __init__(self, f=tanimoto, h=100, rng=None):
        self.rng = rng
        self.h = h
        self.f = f

    def fit(self, X):
        rng = check_random_state(self.rng)
        self.d = X.shape[1]
        self.W = self._select_weights(X, rng)
        self.B = rng.normal(size=self.W.shape[0])*2 + 1e-1
        return self

    def project(self, X):
        return self.f(X, self.W, b=self.B)

    def _select_weights(self, X, rng):
        h = min(self.h, X.shape[0] - 1)
        return X[rng.choice(range(X.shape[0]), size=h, replace=False)]

class ProjectorMixin(object):

    def transform(self, X):
        return self.projector.project(X)

    def project(self, X):
        return self.projector.project(X)


def sigmoid(X, W, b):
    """ Basic sigmoid activation function """
    return 1./(1. + np.exp(X.dot(W.T) - b))

def relu(X, W, b):
    """ Basic rectified linear unit """
    return np.maximum(0, X.dot(W.T) - b)

def tanimoto(X, W, b=None):
    """ Tanimoto similarity function """
    XW = X.dot(W.T)
    XX = np.abs(X).sum(axis=1).reshape((-1, 1))
    WW = np.abs(W).sum(axis=1).reshape((1, -1))
    return XW / (XX+WW-XW)

def sorensen(X, W, b=None):
    """ Sorensen similarity function """
    XW = X.dot(W.T)
    XX = np.abs(X).sum(axis=1).reshape((-1, 1))
    WW = np.abs(W).sum(axis=1).reshape((1, -1))
    return 2 * XW / (XX+WW)


class EEM(object):
    """
    Extreme Entropy Machine

    as presented in
    "Extreme Entropy Machines: Robust information theoretic classification",
    WM Czarnecki and J Tabor,
    Pattern Analysis and Applications (2015)
    DOI: 10.1007/s10044-015-0497-8
    http://link.springer.com/article/10.1007/s10044-015-0497-8
    """

    def __init__(self, h='sqrt', f=tanimoto, C=10000, random_state=None, from_data=True):
        """
        h - number of hidden units, can be
         i) integer, giving the exact number of units
         ii) float, denoting fraction of training set to use (requires: from_data=True)
         iii) string, one of "sqrt", "log", with analogous meaning as the above
        f - activation function (projection)
        C - inverse of covariance estimation smoothing or None for a minimum possible
        from_data - whether to select hidden units from training set (prefered)
        random_state - seed for random number genrator
        """

        self.h = h
        self.C = C
        self.f = f
        self.rs = random_state
        self.fd = from_data
        self._maps = {'sqrt': np.sqrt, 'log': np.log}
        if isinstance(self.h, float):
            if not self.fd:
                raise Exception('Using float as a number of hidden units requires learning from data')
        if isinstance(self.h, str):
            if not self.fd:
                raise Exception('Using string as a number of hidden units requires learning from data')
            if self.h not in self._maps:
                raise Exception(self.h + ' is not supported as a number of hidden units')


    def _pdf(self, x, l):
        """ Returns pdf og l'th class """
        return 1. / np.sqrt(2 * np.pi * self.sigma[l]) * np.exp( - np.power(x - self.m[l], 2) / (2 * self.sigma[l]))

    def _hidden_init(self, X, y):
        """ Initializes hidden layer """
        np.random.seed(self.rs)
        if self.fd:
            if isinstance(self.h, float):
                self.current_h = self.h * X.shape[0]
            elif isinstance(self.h, str):
                self.current_h = self._maps[self.h](X.shape[0])
            else:
                self.current_h = self.h
            self.current_h = max(1, min(self.current_h, X.shape[0]))
            W = X[np.random.choice(range(X.shape[0]), size=self.current_h, replace=False)]
        else:
            self.current_h = self.h
            W = csr_matrix(np.random.rand(self.current_h, X.shape[1]))
        b = np.random.normal(size=self.current_h)
        return W, b

    def fit(self, X, y):
        """ Trains the model """

        self.W, self.b = self._hidden_init(X, y)
        H = self.f(X, self.W, self.b)
        self.labels = np.array([np.min(y), np.max(y)])
        self.m = [0, 0]
        self.sigma = [0, 0]

        for l in range(2):
            data = H[y==self.labels[l]]
            self.m[l] = np.mean(data, axis=0)
            self.sigma[l] = LedoitWolf().fit(data).covariance_
            if self.C is not None:
                self.sigma[l] += np.eye(self.current_h) / (2.0*self.C)

        self.beta = la.pinv(self.sigma[0] + self.sigma[1]).dot((self.m[1] - self.m[0]).T)
        for l in range(2):
            self.m[l] = float(self.beta.T.dot(self.m[l].T))
            self.sigma[l] = float(self.beta.T.dot(self.sigma[l]).dot(self.beta))

    def predict(self, X):
        """ Labels given set of samples """

        p = self.f(X, self.W, self.b).dot(self.beta)
        result = np.argmax([self._pdf(p, l) for l in range(2)], axis=0)
        return self.labels[result]

    def predict_proba(self, X):
        """ Returns probability estimates """

        p = self.f(X, self.W, self.b).dot(self.beta)
        result = np.array([self._pdf(p, l) for l in range(2)]).T
        return result / np.sum(result, axis=1).reshape(-1, 1)
