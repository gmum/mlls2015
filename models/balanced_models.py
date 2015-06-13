import numpy as np
import numpy.linalg as la
from sklearn.preprocessing import LabelBinarizer
import time
from sklearn.svm import SVC
from sklearn.covariance import LedoitWolf as CovEstimator
from sklearn.naive_bayes import GaussianNB
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from misc.config import main_logger
# SVMTAN C [0.001, 10000]
# TWELM/EEM C [1,  100000]
# TWELM/EEM/NB h [100, wielkosc_zbioru_uczacego]

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

class TWELM(ProjectorMixin, BaseEstimator):

    def __str__(self):
        if self.C==None:
            solver =  self.solve.__name__
        else:
            solver = 'algebraic,C='+str(self.C)

        return 'TWELM(h='+str(self.h)+',f='+self.f.__name__+',balanced=true,solver='+solver+',extreme='+str(self.extreme)+')'

    def __init__(self, projector, h=100, C=None, solver=la.lstsq, random_state=0, extreme=True):
        self.h = h
        self.C = C
        self.projector = projector
        self.random_state = random_state
        self.solve = solver

        self.extreme = extreme

    def fit(self, X, y ):
        self.labeler = LabelBinarizer()
        rng = check_random_state(self.random_state)
        self.projector.set_params(h=self.h, rng=rng)
        H = self.projector.fit(X).project(X)

        y = y.tolist()
        s = { l : float(y.count(l)) for l in set(y) }
        ms= max([ s[k] for k in s ])
        s = { l : ms/s[l] for l in s }
        w = np.array( [[ np.sqrt( s[a] ) for a in y ]] ).T

        T = self.labeler.fit_transform(y)
        start = time.time()
        if self.C==None:
            self.beta, _, _, _ = self.solve( np.multiply(H,w), np.multiply(T,w) )
        else:
            H = np.multiply(H,w)
            self.beta = ( la.inv( np.eye(H.shape[1])/self.C + H.T.dot(H) ) ).dot( H.T.dot(np.multiply(T,w)) )

        self.train_time = time.time()-start
        return self

    def predict(self, X ):
        return self.labeler.inverse_transform(np.dot(self.projector.project(X), self.beta)).T

    def decision_function(self, X):
        return np.dot(self.projector.project(X), self.beta)


class RandomNB(ProjectorMixin, BaseEstimator):

    def __str__(self):
        return 'RandNB(h='+str(self.h)+',f='+self.f.__name__+',balanced=true,extreme='+str(self.extreme)+')'

    def __init__(self, projector, h=100, from_data=True, random_state=0,extreme=True):
        self.h=h
        self.from_data = from_data
        self.projector = projector
        self.random_state = random_state
        self.projector.set_params(h=self.h, rng=self.random_state)

        self.extreme = extreme

    def partial_fit(self, X, y):
        rng = check_random_state(self.random_state)
        self.projector.set_params(h=self.h, rng=rng)
        try:
            H = self.projector.project(X)
            self.clf.partial_fit(H, y)
            self.clf.class_prior_ = np.array([0.5, 0.5])
            return self
        except:
            return self.fit(X, y)

    def fit(self, X, y ):
        rng = check_random_state(self.random_state)
        self.projector.set_params(h=self.h, rng=rng)
        H = self.projector.fit(X).project(X)
        self.clf = GaussianNB()
        self.clf.fit(H, y)
        self.clf.class_prior_ = np.array([0.5, 0.5])

        return self

    def predict(self, X ):
        return self.clf.predict(self.projector.project(X))

    def predict_proba(self, X):
        return self.clf.predict_proba(self.projector.project(X)).max(axis=1).reshape(-1, 1)


class SVMTAN(BaseEstimator):

    def __str__(self):
        return 'SVM(kernel=tanimoto, balanced=True, C='+str(self.C)+')'

    def __init__(self, random_state, C=1):
        self.C=C
        self.random_state = random_state

    def fit(self, X, y):
        rng = check_random_state(self.random_state)

        self.clf = SVC(kernel=tanimoto, C=self.C, random_state=rng, class_weight='auto')
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def decision_function(self, X):
        return self.clf.decision_function(X)


class EEM(ProjectorMixin, BaseEstimator):

    def __str__(self):
        return 'EEM(h='+str(self.h)+',f='+self.f.__name__+',C='+str(self.C)+',extreme='+str(self.extreme)+')'

    def __init__(self, projector, h=500, C=None, random_state=0, extreme=True):
        self.h=h
        self.random_state = random_state
        self.projector = projector

        self.C=C

        self.S = [None, None]
        self.m = [None, None]
        self.extreme = extreme


    def _select_weights(self, X, rng):
        h = min(self.h, X.shape[0])
        return X[rng.choice(range(X.shape[0]), size=h, replace=False)]

    def fit(self,X,y):
        rng = check_random_state(self.random_state)
        self.projector.set_params(h=self.h, rng=rng)

        self.neg_label = min(y)
        self.pos_label = max(y)

        y[y==self.neg_label] = -1
        y[y==self.pos_label] = 1


        H = self.projector.fit(X).project(X)

        self.S[0] = CovEstimator(store_precision=False).fit(H[y==-1]).covariance_
        self.S[1] = CovEstimator(store_precision=False).fit(H[y==1]).covariance_

        self.m[0] = H[y==-1].mean(axis=0).T
        self.m[1] = H[y==1].mean(axis=0).T

        if self.C is None:
            S = [self.S[k] for k in range(2)]
        else:
            S = [self.S[k] + 0.5 * (1./self.C) * np.eye(self.S[k].shape[0]) for k in range(2)]

        self.beta = la.inv(S[0] + S[1]).dot(self.m[1]-self.m[0])

        self.proj_mean = [ float(self.beta.T.dot(self.m[k])) for k in range(2)]
        self.proj_var = [ float(self.beta.T.dot(S[k]).dot(self.beta)) for k in range(2)]

        return self

    def Nor(self,x,m,s):

        x = np.array(x.ravel().tolist()[0])
        return 1.0 / (np.sqrt(s) * np.sqrt(2 * np.pi)) * np.exp( -(x-m)**2 / (2*s) )

    def predict(self, X):
        X = self.projector.project(X)
        c0 = self.Nor(X.dot(self.beta),self.proj_mean[0],self.proj_var[0])
        c1 = self.Nor(X.dot(self.beta),self.proj_mean[1],self.proj_var[1])
        return np.array(map(lambda x: self.neg_label if x==-1 else self.pos_label,np.sign(c1-c0)))

    def predict_proba(self, X):
        X = self.projector.project(X)
        c0 = self.Nor(X.dot(self.beta),self.proj_mean[0],self.proj_var[0])
        c1 = self.Nor(X.dot(self.beta),self.proj_mean[1],self.proj_var[1])
        cum = np.vstack((c0, c1)).T
        return (cum /cum.sum(axis=1).reshape(-1,1)).max(axis=1).reshape(-1, 1)

import sys
sys.path.append("..")
import kaggle_ninja
kaggle_ninja.register("EEM", EEM)