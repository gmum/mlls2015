import os
import glob
import logging


import numpy as np
import scipy

from .utils import COMPOUNDS, FINGERPRINTS, split_data, split_data_folds
from .load_data import load_raw_chemical_data, load_meta
from sklearn.utils import check_random_state
from misc.config import DATA_DIR
from misc.cache import cached
from sklearn.base import BaseEstimator
import sklearn

logger = logging.getLogger(__name__)

def identity(X_train, y_train, X_valid, y_valid):
    return (X_train, y_train), (X_valid, y_valid)

def max_abs(X_train, y_train, X_valid, y_valid):
    # Converts [0, 4] --> [0, 4/max(feature)]
    norm = sklearn.preprocessing.MaxAbsScaler()
    X_train = norm.fit_transform(X_train)
    if X_valid.shape[0]:
        X_valid = norm.transform(X_valid)
    return (X_train, y_train), (X_valid, y_valid)


def clip01(X_train, y_train, X_valid, y_valid):
    if isinstance(X_train, scipy.sparse.csr_matrix):
        X_train.data = np.clip(X_train.data, 0, 1)
        X_valid.data = np.clip(X_valid.data, 0, 1)
    else:
        X_train = np.clip(X_train, 0, 1)
        X_valid = np.clip(X_valid, 0, 1)
    return (X_train, y_train), (X_valid, y_valid)


@cached
def calculate_jaccard_kernel(data, fold):
    (X_train, y_train), (X_valid, y_valid) = data.get_data(fold=fold)

    assert data.get_params()['preprocess'] == "max_abs" or data.get_params()['preprocess'] == "clip01", \
        "Dataset should be normalized"

    def _calculate_jaccard_kernel(X1T, X2T):
        X1T_sums = np.array(X1T.sum(axis=1))
        X2T_sums = np.array(X2T.sum(axis=1))
        K = X1T.dot(X2T.T)

        if hasattr(K, "toarray"):
            K = K.toarray()

        K2 = -(K.copy())
        K2 += (X1T_sums.reshape(-1, 1))
        K2 += (X2T_sums.reshape(1, -1))
        K = K / K2
        return K

    return _calculate_jaccard_kernel(X_train, X_train), _calculate_jaccard_kernel(X_valid, X_train)

def assign_cluster_id(data, fold, cluster_files):
    """ Assign each point closest cluster

        Parameters
        ----------
        cluster_files: list
          List of files to intersect with


        Returns (X_train_inside, X_train_cluster_id, y_train), (X_valid_inside, X_valid_cluster_id, y_valid)
    """
    pass

class BaseChemDataset(BaseEstimator):
    """
    Base dataset for chemical compounds. Accepts various representations for chemical compound
    """

    def __init__(self, compound, representation, valid_size, rng=777, preprocess="", n_features=None):
        self.compound = compound
        self.n_features = n_features
        self.representation = representation
        self.rng = rng
        self.preprocess = preprocess
        self.valid_size = valid_size
        if not isinstance(self.rng, int):
            logger.warning("Passed rng as RandomState, will result in different folds every time get_train_data"
                           "is called.")

        self._check_validity()

    def get_data(self):
        X, y = load_raw_chemical_data(self.compound, representation=self.representation, n_features=self.n_features)
        # rng = check_random_state(self.rng)

        if self.preprocess:
            if self.preprocess not in globals():
                raise RuntimeError("Not found preprocess fnc")

            pre = globals()[self.preprocess]
        else:
            pre = identity

        if self.valid_size == 0:
            return pre(X, y, np.empty(shape=(0, X.shape[1])), np.empty(shape=(0,)))
        else:
            train, valid = split_data(X, y, rng=self.rng, test_size=self.valid_size)
            return pre(train[0], train[1], valid[0], valid[1])

    def _check_validity(self):
        assert self.compound in COMPOUNDS or self.compound.split("_")[0] in COMPOUNDS
        # assert self.representation in (FINGERPRINTS + ['smiles', "RAWSRMACCS"])
        assert isinstance(self.rng, int) or isinstance(self.rng, np.random.RandomState) or self.rng is None
        assert self.valid_size >= 0

    def _get_preprocess(self):
        if self.preprocess:
            if self.preprocess not in globals():
                raise RuntimeError("Not found preprocess fnc")

            return globals()[self.preprocess]
        else:
            return identity
        

class CVBaseChemDataset(BaseChemDataset):
    """
    Base dataset for chemical compounds
    """

    def __init__(self, compound, representation, n_folds, rng=777, preprocess="", n_features=None):
        super(CVBaseChemDataset, self).__init__(compound=compound, representation=representation,
                                                valid_size=0, rng=rng, preprocess=preprocess, n_features=n_features)
        self.n_folds = n_folds

    def _get_fold(self, fold):
        assert 0 <= fold < self.n_folds, "Fold should be in range [0, n_folds - 1]"

        X, y = load_raw_chemical_data(self.compound, representation=self.representation, n_features=self.n_features)
        return split_data_folds(X, y, rng=self.rng, n_folds=self.n_folds, fold=fold)

    def get_meta(self, key, fold):
        train, valid, ids  = self._get_fold(fold=fold)
        meta = load_meta(compound=self.compound, representation=self.representation)
        logger.info("Loaded meta with keys " + str(meta.keys()))
        if key not in meta:
            raise RuntimeError("Not found requested key in meta file")
        return meta[key][ids[0]], meta[key][ids[1]]

    def get_data(self, fold=0):
        pre = self._get_preprocess()
        train, valid, _  = self._get_fold(fold=fold)
        return pre(train[0], train[1], valid[0], valid[1])