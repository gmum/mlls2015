#TODO: comapre to new version from ECML competition code

from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import StratifiedShuffleSplit, StratifiedKFold
import scipy
import scipy.stats
from itertools import product
from copy import copy

from misc.config import main_logger, c
from kaggle_ninja.cached import *
kaggle_ninja.setup_ninja(logger=main_logger, cache_dir=c["CACHE_DIR"])


def get_data(loader, preprocess_fncs):
    """
    Function for loading data for multiple compounds and fingerprints
    :param loader: tuple, loader function and its parameters
    :param preprocess_fncs: tuple, preprocess function and it's parameters
    :return: list of data for all compound and fingerprint combinations
    """
    proteins = loader[1]['protein']
    fingerprints = loader[1]['fingerprint']

    data = {}

    for pair in product(proteins, fingerprints):
        single_loader = copy(loader)
        single_loader.update({'protein': pair[0], 'fingerprint': pair[1]})
        compound_data = _get_single_data(loader, preprocess_fncs)
        data[pair[0] + "_" + pair[1]] = compound_data

    return data

@cached(save_fnc=joblib_save, load_fnc=joblib_load, check_fnc=joblib_check, cached_ram=True)
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

    test_data[0].data.setflags(write = False) # X
    test_data[1].setflags(write = False)      # Y

    # Note: no preprocessing done on X_test, Y_test, we have to do it manually for now
    return folds, test_data



### Raw data ###

def _get_raw_data(compound, fingerprint):
    file_name = os.path.join(c["DATA_DIR"], compound + "_" + fingerprint + ".libsvm")
    assert os.path.exists(file_name)
    X, y = load_svmlight_file(file_name)
    return X.toarray(), y


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

    if test_size > 0:
        split_indices = StratifiedShuffleSplit(y, n_iter=1, test_size=test_size, random_state=seed)
        assert len(split_indices) == 1
        for train_index, test_index in split_indices:
            X_test, y_test = X[test_index], y[test_index]
            X, y = X[train_index], y[train_index]

    fold_indices = StratifiedKFold(y, n_folds=n_folds, shuffle=True, random_state=seed)

    folds = []
    for train_index, valid_index in fold_indices:
        folds.append({'X_train': X[train_index],
                      'Y_train': y[train_index],
                      'X_valid': X[valid_index],
                      'Y_valid': y[valid_index]})

    return folds, [(X_test, y_test)]

#### Preprocess functions ####

def bucket_simple_threshold(fold, threshold_bucket, others_to_preprocess=None, normalize=False, implicative_ones=False):
    """
    @param implicative_ones If bucket i is 1 then i-1...0 are 1
    """
    X_train, Y_train, X_valid, Y_valid = \
        fold["X_train"], fold["Y_train"], fold["X_valid"], fold["Y_valid"]

    ranges_max = [X_train[:,i].max() for i in range(X_train.shape[1])]
    ranges_min = [X_train[:,i].min() for i in range(X_train.shape[1])]
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]
    X_train_transposed = X_train.T.tocsr() # Just for speed!
    # This line of code gets **for each feature(column)** counts of values
    freq = [\
            [(X_train_transposed[i,:]==s).sum() for s in xrange(int(ranges_min[i]), int(ranges_max[i])+1)]\
            for i in range(n_features) \
           ]

    # We encode value as hot one IFF it will be active for at least threshold_bucket% times
    # else we will just merge it with next one
    # We output mapping for each column to end index (0/1)
    mapping = []
    for i in xrange(n_features):
        mapping.append({})
    output_column = 0
    for id_feature, freq_list in enumerate(freq):
        output_column_count = [0]
        # For feature go through possible values
        column_range =  int(ranges_max[id_feature]) - int(ranges_min[id_feature])
        # Without 0!
        for id, target_value in enumerate(range(1, column_range+1)):
            target_value += ranges_min[id_feature]
            mapping[id_feature][target_value] = output_column
            output_column_count[-1] += freq_list[id]
            if output_column_count[-1] > float(n_samples)*threshold_bucket and id != column_range - 1:
                output_column_count.append(0)
                output_column += 1
        # Might have associated last output without reason
        if column_range != 0 and output_column_count[-1] < float(n_samples)*threshold_bucket:
            output_column -= 1
            mapping[id_feature][target_value] -= 1 # Useful for the first time in history (scope sucks in python 2.7)

        # Because we are now going to the next row
        output_column += 1

    # Create new matrix row by row TODO: Y is not used here, remove?
    def bucket(X, Y=None):
        col_ptr = []
        data = []
        row_ind = [0]

        counter = 0
        current_data_ind = 0 # not used
        for i in range(X.shape[0]):
            row = [] # not used
            for col_idx, col_val in enumerate(X[i,:].toarray().reshape(-1)):
                # Faster than iterating X[i,:].T, iterating scipy sparse is slow
                if col_val != 0 and len(mapping[col_idx]):
                    # Map to calculated bucket, or maximal bucket for this column
                    if implicative_ones:
                        for val in xrange(1, int(col_val)+1):
                            col_ptr.append(mapping[col_idx].get(int(val), max(mapping[col_idx].values())))
                            data.append(1.0 if normalize else 1) # dtype = int
                            counter += 1
                    else:
                        col_ptr.append(mapping[col_idx].get(int(col_val), max(mapping[col_idx].values())))
                        data.append(1.0 if normalize else 1)
                        counter += 1

            row_ind.append(counter)

        return scipy.sparse.csr_matrix((np.array(data), np.array(col_ptr), \
                                         np.array(row_ind)), shape=(X.shape[0], output_column))



    fold["X_train"] = bucket(X_train, Y_train)
    scaler = StandardScaler(with_mean=False, copy=False)

    if normalize:
        fold["X_train"] = scaler.fit_transform(fold["X_train"])

    if X_valid.shape[0]:
        fold["X_valid"] = bucket(X_valid, Y_valid)
        if normalize:
            fold["X_valid"] = scaler.transform(fold["X_valid"])

    test_data = []
    for (X, Y) in others_to_preprocess:
        if X.shape[0]:
            test_data.append([bucket(X, Y), Y])
            if normalize:
                test_data[-1][0] = scaler.transform(X)
    return fold, test_data

import kaggle_ninja
kaggle_ninja.register("get_splitted_data", get_splitted_data)
kaggle_ninja.register("bucket_simple_threshold", bucket_simple_threshold)
