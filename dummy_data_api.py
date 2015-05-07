import os
import itertools
import numpy as np
import scipy
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import train_test_split, StratifiedKFold

from misc.config import c

data_dir = c['DATA_DIR']


def get_data(data_desc):
    X, Y = load_svmlight_file(data_desc['file'])
    tr_ind, ts_ind = train_test_split(X, test_size=data_desc['p_test'])

    tr_X = X[tr_ind]
    tr_Y = Y[tr_ind]

    ts_X = X[ts_ind]
    ts_Y = Y[ts_ind]

    folds_ind = StratifiedKFold(tr_Y, n_folds=data_desc['n_folds'])
    folds = []
    for tr_ind, ts_ind in folds_ind:
        folds.append([tr_X[tr_ind], tr_Y[tr_X], tr_X[ts_ind], tr_Y[ts_ind]])

    return folds, [ts_X, ts_Y]


def get_default_data_desc(comp='5ht2a', fp='EstateFP'):
    file_name = os.path.join(data_dir, comp + "_" + fp + ".libsvm")
    assert os.path.exists(file_name)

    data_desc = {
        'file': file_name,
        'p_test': 0.1,
        'preprocess': None,
        'seed': 666,
        'n_folds': 5
    }

    return data_desc


def set_representation_by_buckets(X):
    """
    Helper function transforms X into bucketed representation with features [x[j]>=bucket[j,i] for i,j in .. ]
    """

    # Investigate columns ranges
    ranges_max = X.max(axis=0).toarray().flatten()
    ranges_min = X.min(axis=0).toarray().flatten()
    spreads = [mx - mn for mx, mn in itertools.izip(ranges_max, ranges_min)]

    bucket_count = min(max(spreads), 5)

    # Iterate each column and create evenly spread buckets
    buckets = []
    for col in X.T:
        col = np.array(col.toarray().reshape(-1))
        col.sort()
        col = col[np.where(col>0)[0][0]:]
        col_buckets = [a[0] for a in np.array_split(col, min(bucket_count, len(col)))]
        if len(col_buckets) < bucket_count:
            col_buckets += [col_buckets[-1]]*-(len(col_buckets)-bucket_count)
        buckets.append(col_buckets)

    # Create new matrix row by row
    feature_dict = {}

    X_tr = np.zeros(shape=(X.shape[0],bucket_count*X.shape[1]), dtype="int32")
    for i in range(X.shape[0]):
        row = []
        for col_idx, col in enumerate(X[i,:].toarray().reshape(-1)):
            # Faster than iterating X[i,:].T, iterating scipy sparse is slow
            for b,x in enumerate(buckets[col_idx]):
                assert(x <= ranges_max[col_idx])
                if(col >= x):
                    f = str(col_idx)+">= bucket_"+str(b)
                    if f not in feature_dict:
                        feature_dict[f] = len(feature_dict)
                    row.append(feature_dict[f])

        X_tr[i, row] = 1

    return scipy.sparse.csr_matrix(X_tr)