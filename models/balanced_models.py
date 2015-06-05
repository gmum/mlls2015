import numpy as np
import numpy.linalg as la
from sklearn.preprocessing import LabelBinarizer
import time
from sklearn.svm import SVC
from sklearn.covariance import LedoitWolf as CovEstimator
from sklearn.naive_bayes import GaussianNB
from sklearn.base import BaseEstimator

# SVM C [0.001, 10000]
# ELM/EEM C [1, 100000]
# ELM/EEM/NB h [100, wielkosc_zbioru_uczacego]

def sigmoid(x,w,b):
    return 1/(1+np.exp(-(np.dot(x,w)+b)))
 
def rbf(X,W,b=1):
    XW = np.dot(X, W.T)
    XX = (X ** 2).sum(axis=1).reshape(-1, 1)
    WW = (W ** 2).sum(axis=1).reshape(1, -1)
    return  np.exp( - np.multiply(b, -2*XW + XX + WW) )
 
def tanimoto(X, W, b=None):
    XW = X.dot(W.T)
    XX = X.multiply(X).sum(axis=1).reshape(-1, 1)
    WW = W.multiply(W).sum(axis=1).reshape(1, -1)
    return XW / (XX+WW-XW)
 
def identity(X, W, b):
    return X
 
def BAC(y_pred, y_true):
    return np.mean([ sum(1. for label, pred in zip(y_true, y_pred) if label == pred and label == y)/sum(1. for label in y_true if label == y) for y in set(y_true)])
 

class RandomProjectionMixin(object):
 
    def _select_weights(self, X):
        h = min(self.h, X.shape[0])
        return X[np.random.choice(range(X.shape[0]), size=h, replace=False)]
 
    def _fit_projection(self,X):
        np.random.seed(self.random_state)
        self.d = X.shape[1]
        self.W = self._select_weights(X)
        self.B = np.random.normal(size=self.W.shape[0])
 
    def _project(self,X):
        if self.extreme:
            return self.f(X, self.W, self.B)
        else:
            P = self.f(X, self.W, self.B)
            # WARNING: Densifying data!
            R = X.toarray()
            return np.hstack((P, R))


class TWELM(RandomProjectionMixin, BaseEstimator):
 
    def __str__(self):
        if self.C==None:
            solver =  self.solve.__name__ 
        else:
            solver = 'algebraic,C='+str(self.C)
 
        return 'TWELM(h='+str(self.h)+',f='+self.f.__name__+',balanced=true,solver='+solver+',extreme='+str(self.extreme)+')'
 
    def __init__(self, h=100, f=tanimoto, C=None, solver=la.lstsq, random_state=0, extreme=True):
        self.h=h
        self.f=f
        self.C=C
        self.labeler = LabelBinarizer()
        self.solve = solver
        self.random_state = random_state
        self.extreme = extreme
 
    def fit(self, X, y ):
       
        self._fit_projection(X)
        H = self._project(X)
 
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
            #HW = np.multiply(H,w**2)
            H = np.multiply(H,w)            
            self.beta = ( la.inv( np.eye(H.shape[1])/self.C + H.T.dot(H) ) ).dot( H.T.dot(np.multiply(T,w)) ) 
 
        self.train_time = time.time()-start
        return self
        
    def predict(self, X ):
        return self.labeler.inverse_transform(np.dot(self._project(X), self.beta)).T
 
    def decision_function(self, X):
        return np.dot(self._project(X), self.beta)


class RandomNB(RandomProjectionMixin, BaseEstimator):
 
    def __str__(self):
        return 'RandNB(h='+str(self.h)+',f='+self.f.__name__+',balanced=true,extreme='+str(self.extreme)+')'
 
    def __init__(self, h=100, f=tanimoto, from_data=True, random_state=0,extreme=True):
        self.h=h
        self.f=f
        self.from_data = from_data
        self.random_state = random_state
        self.extreme = extreme
 
    def partial_fit(self, X, y):
        try:
            H = self._project(X)
            self.clf.partial_fit(H, y)
            self.clf.class_prior_ = np.array([0.5, 0.5])
            return self
        except:
            return self.fit(X, y)
 
    def fit(self, X, y ):
 
        self._fit_projection(X)
        H = self._project(X)
        self.clf = GaussianNB()
        self.clf.fit(H, y)
        self.clf.class_prior_ = np.array([0.5, 0.5])
 
        return self

    def predict(self, X ):
        return self.clf.predict(self._project(X))
 
    def predict_proba(self, X):
        return self.clf.predict_proba(self._project(X)).max(axis=1).reshape(-1, 1)
 
 
class SVMTAN(BaseEstimator):
 
    def __str__(self):
        return 'SVM(kernel=tanimoto, balanced=True, C='+str(self.C)+')'
 
    def __init__(self, random_state, C=1):
        self.C=C
 
    def fit(self, X, y):
        self.clf = SVC(kernel=tanimoto, C=self.C, class_weight='auto')
        self.clf.fit(X, y)
        return self
 
    def predict(self, X):
        return self.clf.predict(X)
 
    def decision_function(self, X):
        return self.clf.decision_function(X)

 
class EEM(RandomProjectionMixin, BaseEstimator):
 
    def __str__(self):
        return 'EEM(h='+str(self.h)+',f='+self.f.__name__+',C='+str(self.C)+',extreme='+str(self.extreme)+')'
 
    def __init__(self, h=400, f=tanimoto, C=None, random_state=0, extreme=True):
        self.h=h
        self.f=f
        self.C=C
        self.random_state = random_state
        self.S = [None, None]
        self.m = [None, None]
        self.extreme = extreme
 
 
    def _select_weights(self, X):
        h = min(self.h, X.shape[0])
        return X[np.random.choice(range(X.shape[0]), size=h, replace=False)]
 
    def fit(self,X,y):
 
        self.neg_label = min(y)
        self.pos_label = max(y)
 
        y[y==self.neg_label] = -1
        y[y==self.pos_label] = 1
 
 
        self._fit_projection(X)
        H = self._project(X)
 
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
 
    def Nor(self,x,m,s):
 
        x = np.array(x.ravel().tolist()[0])
        return 1.0 / (np.sqrt(s) * np.sqrt(2 * np.pi)) * np.exp( -(x-m)**2 / (2*s) )
 
    def predict(self, X):
        X = self._project(X)
        c0 = self.Nor(X.dot(self.beta),self.proj_mean[0],self.proj_var[0])
        c1 = self.Nor(X.dot(self.beta),self.proj_mean[1],self.proj_var[1])
        return np.array(map(lambda x: self.neg_label if x==-1 else self.pos_label,np.sign(c1-c0)))
 
    def predict_proba(self, X):
        X = self._project(X)
        c0 = self.Nor(X.dot(self.beta),self.proj_mean[0],self.proj_var[0])
        c1 = self.Nor(X.dot(self.beta),self.proj_mean[1],self.proj_var[1])
        cum = np.vstack((c0, c1)).T
        return (cum /cum.sum(axis=1).reshape(-1,1)).max(axis=1).reshape(-1, 1)