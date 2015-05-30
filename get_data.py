#TODO: Make passing n_folds=1 possible
import sys
sys.path.append("..")
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import StratifiedShuffleSplit, StratifiedKFold
import scipy
import scipy.stats
from copy import copy

from misc.config import main_logger, c
import kaggle_ninja
from kaggle_ninja.cached import *

if c["USE_GC"]:
    kaggle_ninja.setup_ninja(logger=main_logger, google_cloud_cache_dir="gs://al_ecml/cache", cache_dir=c["CACHE_DIR"])
else:
    kaggle_ninja.setup_ninja(logger=main_logger, cache_dir=c["CACHE_DIR"])

import logging
# main_logger.setLevel(logging.DEBUG)

def get_data(compounds, loader, preprocess_fncs):
    """
    Function for loading data for multiple compounds and fingerprints
    :param loader: tuple, loader function and its parameters
    :param preprocess_fncs: tuple, preprocess function and it's parameters
    :return: list of data for all compound and fingerprint combinations
    """

    ret = {}
    for pair in compounds:
        single_loader = copy(loader)
        single_loader[1].update({'compound': pair[0], 'fingerprint': pair[1]})
        data_desc = {'loader': single_loader, 'preprocess_fncs': preprocess_fncs}
        compound_data = _get_single_data(**data_desc)
        ret[pair[0] + "_" + pair[1]] = compound_data

    return ret

@cached(save_fnc=joblib_save, load_fnc=joblib_load, check_fnc=joblib_check, cached_ram=False)
def _get_single_data(loader, preprocess_fncs):
    # Load

    loading_fnc = find_obj(loader[0])
    folds, test_data = loading_fnc(**loader[1])

    # Run preprocessing !
    for id, f in enumerate(folds):
        main_logger.info("Running preprocess on "+str(id)+" fold")
        for prep in preprocess_fncs:
            preprocess_fnc = find_obj(prep[0])
            if id == 0: # Hack - first preprocess runs also preprocessing on the rest of datasets
                folds[id], test_data = preprocess_fnc(f, others_to_preprocess=test_data, **prep[1])
            else:
                folds[id], _ = preprocess_fnc(f, **prep[1])

    # Ensure no one modifies it later on
    for f in folds:
        f["X_train"].data.setflags(write = False)
        f["Y_train"].setflags(write = False)
        f["X_valid"].data.setflags(write = False)
        f["Y_valid"].setflags(write = False)



    assert len(test_data) <= 2

    if len(test_data) > 0:
        test_data[0].data.setflags(write = False) # X
        test_data[1].setflags(write = False)      # Y

    data_desc = {'loader': loader, 'preprocess': preprocess_fncs}

    return [folds, test_data, data_desc]


### Raw data ###

def _get_raw_data(compound, fingerprint):
    file_name = os.path.join(c["DATA_DIR"], compound + "_" + fingerprint + ".libsvm")
    assert os.path.exists(file_name)
    X, y = load_svmlight_file(file_name)
    return X, y


@cached(save_fnc=joblib_save, load_fnc=joblib_load, check_fnc=joblib_check)
def get_splitted_data(compound, fingerprint, n_folds, seed, test_size=0.0):
    """
    Returns data of given compound docked as given fingerprint as folds with training
    and validating data and separate test data

    :param compound desired compound
    :param fingerprint desired fingerprint
    :param n_folds number of folds in train/valid data
    :param test_size test dataset (final validation) is 0.1*100% * number of examples
    """
    X, y = _get_raw_data(compound, fingerprint)
    test_data = []
    if test_size > 0:
        split_indices = StratifiedShuffleSplit(y, n_iter=1, test_size=test_size, random_state=seed)
        assert len(split_indices) == 1
        for train_index, test_index in split_indices:
            X_test, y_test = X[test_index], y[test_index]
            X, y = X[train_index], y[train_index]
        test_data = (X_test, y_test)

    #TODO: Make passing n_folds=1 possible
    fold_indices = StratifiedKFold(y, n_folds=n_folds, shuffle=True, random_state=seed)

    folds = []
    for train_index, valid_index in fold_indices:
        folds.append({'X_train': (X[train_index]).copy(),
                      'Y_train': (y[train_index]).copy(),
                      'X_valid': (X[valid_index]).copy(),
                      'Y_valid': (y[valid_index]).copy()})

    return folds, test_data

#### Preprocess functions ####
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
def to_binary(fold, others_to_preprocess=[], threshold_bucket=0, all_below=False):
    """
    @param implicative_ones If bucket i is 1 then i-1...0 are 1
    """
    X_train, Y_train, X_valid, Y_valid = \
        fold["X_train"].astype("int32"), fold["Y_train"].astype("int32"), fold["X_valid"].astype("int32"), fold["Y_valid"].astype("int32")

    transformer = DictVectorizer(sparse=True)

    def to_dict_values(X):
        dicted_rows = []
        frequencies = defaultdict(int)
        for i in xrange(X.shape[0]):
            dicted_rows.append({})
        for id, row in enumerate(X):
            for column, value in zip(row.indices, row.data):
                if all_below:
                    for value_iterative in range(value+1):
                        dicted_rows[id][str(column)+"="+str(value_iterative)] = 1
                        frequencies[str(column)+"="+str(value_iterative)] += 1
                else:
                    dicted_rows[id][str(column)+"="+str(value)] = 1
                    frequencies[str(column)+"="+str(value)] += 1
        return dicted_rows, frequencies

    D, freqs = to_dict_values(X_train)
    fold["X_train"] = transformer.fit_transform(D)

    if X_valid.shape[0]:
        fold["X_valid"] = transformer.transform(to_dict_values(X_valid)[0])

    # Wychodzi 0 dla valid and test
    test_data = []
    if len(others_to_preprocess):
        X = others_to_preprocess[0]
        Y = others_to_preprocess[1]
        if X.shape[0]:
            D, _ = to_dict_values(X.astype("int32"))
            test_data = [transformer.transform(D), Y]
        else:
            test_data = [X.astype("int32"),Y]
    return fold, test_data


import kaggle_ninja
kaggle_ninja.register("get_splitted_data", get_splitted_data)
kaggle_ninja.register("to_binary", to_binary)
