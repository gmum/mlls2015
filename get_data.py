#TODO: More general pipepline for PLOS publication
#TODO: add valid size if, so that we do not need n_folds==0
import sys
sys.path.append("..")
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import StratifiedShuffleSplit, StratifiedKFold
import scipy
import scipy.stats
import copy
from models.strategy import *
from misc.config import main_logger, c
import kaggle_ninja
from kaggle_ninja.cached import *
import logging
from sklearn.metrics import pairwise_distances
if c["USE_GC"]:
    kaggle_ninja.setup_ninja(logger=main_logger, google_cloud_cache_dir="gs://al_ecml/cache", cache_dir=c["CACHE_DIR"])
else:
    kaggle_ninja.setup_ninja(logger=main_logger, cache_dir=c["CACHE_DIR"])

import logging
# main_logger.setLevel(logging.DEBUG)

fingerprints = ["EstateFP", "ExtFP", "KlekFP", "KlekFPCount", "MACCSFP", "PubchemFP", "SubFP", "SubFPCount"]
proteins = ['5ht7','5ht6','SERT','5ht2c','5ht2a','hiv_integrase','h1','hERG','cathepsin','hiv_protease','M1','d2']


@cached()
def test_cache(x):
    return x*3



from sklearn.preprocessing import scale

from models.balanced_models import *

def get_data_by_name(loader, preprocess_fncs, name):
    """
    :param name: X_train.idx, X_valid.idx, Y_train.idx, X_train_test, X_valid, X_test
    """
    loader = copy.deepcopy(loader)
    preprocess_fncs= copy.deepcopy(preprocess_fncs
    )
    name_splitted, idx = name.split(".")

    # This is hacky, but ensures that we don't recalculate get_single_data

    compounds = [[loader[1]['compound'], loader[1]['fingerprint']]]
    del loader[1]['compound']
    del loader[1]['fingerprint']

    folds, _, _ = get_data(compounds=compounds, loader=loader, preprocess_fncs=preprocess_fncs).values()[0]
    return folds[int(idx)][name_splitted]

@cached(cached_ram=True)
def get_tanimoto_projection(loader, preprocess_fncs, seed, name, h=100):
    X = get_data_by_name(loader, preprocess_fncs, name)["data"]
    m = RandomProjector(f=tanimoto, h=h, rng=seed).fit(X)
    return scale(m.predict(X), with_mean=True, with_std=True)*0.5 # 0.5 std :)


@cached(cached_ram=True)
def get_tanimoto_pairwise_distances(loader, preprocess_fncs, name):
    X = get_data_by_name(loader, preprocess_fncs, name)["data"]
    X_dist = pairwise_distances(X.astype("float32"), metric=jaccard_dist)
    if hasattr(X_dist, "toarray"):
        X_dist = X_dist.toarray()
    return X_dist


def get_data(compounds, loader, preprocess_fncs):
    """
    Function for loading data for multiple compounds and fingerprints
    :param loader: tuple, loader function and its parameters
    :param preprocess_fncs: tuple, preprocess function and it's parameters
    :return: list of data for all compound and fingerprint combinations
    """

    assert "compound" not in loader and "fingerprint" not in loader and\
        "seed" not in loader \
        , "Please don't pass compound/fingerprint/seed in loader!"

    ret = {}
    for pair in compounds:
        #TODO: brzydki kod
        single_loader = copy.deepcopy(loader)
        single_loader[1].update({'compound': pair[0], 'fingerprint': pair[1]})
        data_desc = {'loader': single_loader, 'preprocess_fncs': preprocess_fncs}
        compound_data = _get_single_data(**data_desc)
        ret[pair[0] + "_" + pair[1]] = compound_data

    return ret

@cached(save_fnc=joblib_save, load_fnc=joblib_load, check_fnc=joblib_check, cached_ram=True)
def _get_single_data(loader, preprocess_fncs):
    # Load

    loading_fnc = find_obj(loader[0])
    folds, test_data = loading_fnc(preprocess_fncs=preprocess_fncs, **loader[1])

    # Run preprocessing !
    for id, f in enumerate(folds):
        main_logger.info("Running preprocess on "+str(id)+" fold")
        for prep in preprocess_fncs:
            preprocess_fnc = find_obj(prep[0])
            if id == 0: # Hack - first preprocess runs also preprocessing on the rest of datasets
                folds[id], test_data = preprocess_fnc(f, others_to_preprocess=test_data, **prep[1])
            else:
                folds[id], _ = preprocess_fnc(f, **prep[1])


    try:
        # Ensure no one modifies it later on
        for f in folds:
            f["X_train"]["data"].data.setflags(write = False)
            f["Y_train"]["data"].setflags(write = False)
            f["X_valid"]["data"].data.setflags(write = False)
            f["Y_valid"]["data"].setflags(write = False)
    except:
        main_logger.warning("Wasn't able to set write/read flags")


    assert len(test_data) <= 2

    if len(test_data) > 0:
        test_data[0]["X"]["data"].data.setflags(write = False) # X
        test_data[0]["Y"]["data"].setflags(write = False)      # Y

    data_desc = {'loader': loader, 'preprocess': preprocess_fncs}


    for source in [folds, test_data]:
        for f in source:
            for dataset in f:
                f[dataset]["i"]["name"] = dataset + "." + str(f[dataset]["i"]["id"])
                f[dataset]["i"]["loader"] = loader
                f[dataset]["i"]["preprocess_fncs"] = preprocess_fncs

    return [folds, test_data, data_desc]


### Raw data ###

def _get_raw_data(compound, fingerprint):
    file_name = os.path.join(c["DATA_DIR"], compound + "_" + fingerprint + ".libsvm")
    assert os.path.exists(file_name)
    X, y = load_svmlight_file(file_name)
    return X, y


def get_splitted_data_checkerboard(compound, fingerprint, n_folds, seed,valid_size, preprocess_fncs=None,  test_size=0.0):
    X = np.random.uniform(-1, 1, size=(10000,2))
    positive_quadrant=X[(X[:,0]>0) & (X[:,1]>0),:]
    negative_quadrant=X[(X[:,0]<0) & (X[:,1]<0),:]
    X = np.vstack([positive_quadrant, negative_quadrant])
    Y = np.ones(shape=(X.shape[0], ))
    Y[0:positive_quadrant.shape[0]] = -1
    return _split(X, Y, n_folds=n_folds, seed=seed, test_size=test_size, valid_size=valid_size)


def get_splitted_uniform_data(compound, fingerprint, n_folds, seed,valid_size, preprocess_fncs=None, test_size=0.0):
    X = np.random.uniform(-1, 1, size=(3000, 2))
    y = np.ones(X.shape[0])
    negative_examples = np.where(X[:, 0] < 0)
    y[negative_examples] = -1
    return _split(X, y, n_folds=n_folds, seed=seed, test_size=test_size, valid_size=valid_size)


from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import jaccard
from itertools import product

def calculate_jaccard_distance(protein, fingerprint, seed, preprocess_fncs, valid_size, only_positive=False):
    loader = ["get_splitted_data",
              {"n_folds": 0,
               "valid_size": valid_size,
               "seed":seed,
               "test_size":0.0}]

    data = get_data([[protein, fingerprint]], loader, preprocess_fncs)


    X1T = data[protein+"_"+fingerprint][0][0]["X_train"]["data"]

    if only_positive:
        Y = data[protein+"_"+fingerprint][0][0]["Y_train"]["data"]
        X1T = X1T[Y==1]
    X2T = X1T
    X1T_sums = np.array(X1T.sum(axis=1))
    X2T_sums = np.array(X2T.sum(axis=1))
    K = X1T.dot(X2T.T)
    K = K.toarray()
    K2 = -(K.copy())
    K2 += (X1T_sums.reshape(-1,1))
    K2 += (X2T_sums.reshape(1,-1))
    K = K/K2
    return X1T, 1 - K

def calculate_jaccard_kernel(X1T, X2T):
    X1T_sums = np.array(X1T.sum(axis=1))
    X2T_sums = np.array(X2T.sum(axis=1))
    K = X1T.dot(X2T.T)
    K = K.toarray()
    K2 = -(K.copy())
    K2 += (X1T_sums.reshape(-1,1))
    K2 += (X2T_sums.reshape(1,-1))
    K = K/K2
    return X1T, 1 - K

def interestingness_index(X, subtree_a, subtree_b):
    mean_subtree_a = calculate_jaccard_kernel(X[subtree_a], X[subtree_a])[1].mean()
    mean_subtree_b = calculate_jaccard_kernel(X[subtree_b], X[subtree_b])[1].mean()
    mean_inter = calculate_jaccard_kernel(X[subtree_a], X[subtree_b])[1].mean()
    return mean_inter/mean_subtree_a + mean_inter/mean_subtree_b

def jaccard_distance_index(subtree_a, subtree_b):
    start = min(min(subtree_a), min(subtree_b))
    ind_rage = range(start, max(max(subtree_a), max(subtree_b))+1)
    A = [1 if i in subtree_a else 0 for i in ind_rage]
    B = [1 if i in subtree_b else 0 for i in ind_rage]
    return 1-jaccard(A, B)


@cached(save_fnc=joblib_save, load_fnc=joblib_load, check_fnc=joblib_check, cached_ram=True)
def get_splitted_data_clusterwise_Sabina(compound, fingerprint, seed, preprocess_fncs, valid_size, n_folds, test_size=0.0):
    pass


def _generate_fold_indices(y, valid_size, seed, n_folds):
    folds = []
    for i in range(n_folds):
        folds.append(list(StratifiedShuffleSplit(y, n_iter=10, test_size=0.15, random_state=seed+i))[0])
    return folds

@cached(save_fnc=joblib_save, load_fnc=joblib_load, check_fnc=joblib_check, cached_ram=True)
def get_splitted_data_clusterwise(compound, fingerprint, seed, preprocess_fncs, n_folds, \
                                  valid_size,
                                  test_size=0.0,
                                  cluster_size_threshold_A=0.42, \
                                  cluster_size_threshold=0.1):
    main_logger.info("Calculating jaccard distances")
    X, K = calculate_jaccard_distance(protein=compound, fingerprint=fingerprint, \
                                      seed=seed, valid_size=valid_size, preprocess_fncs=preprocess_fncs, only_positive=False)

    assert(X.shape[0] == K.shape[0])
    ## Fit aglomerative clustering to array of pairwise distances
    m = AgglomerativeClustering(n_clusters=2, \
                                linkage='complete',
                                affinity="precomputed")
    main_logger.info("Fitting AgglomerativeClustering")
    m.fit_predict(K)

    ## Calculate sufficiently big subtrees
    id_to_nodes = {i: [i] for i in range(K.shape[0])}
    check_threshold = int(cluster_size_threshold * K.shape[0])
    check_threshold_A = int(cluster_size_threshold_A * K.shape[0])

    # Nodes in m.children_ are sorted by merging time
    for id, n in enumerate(m.children_):
        id_to_nodes[id  + K.shape[0]] = id_to_nodes[n[0]] + id_to_nodes[n[1]]

    big_subtrees = [id_to_nodes[i] for i in id_to_nodes if len(id_to_nodes[i]) > check_threshold]

    ## Calculate clusters and pick pair with the biggest difference
    main_logger.info("Calculating interestingness index for "+str(len(big_subtrees)**2)+" pairs")
    pair = []
    last_log_id_1 = -1
    for (id_1, t_1), (id_2, t_2) in product(enumerate(big_subtrees), enumerate(big_subtrees)):
        if id_1 != last_log_id_1:
            main_logger.info("id_1: "+str(id_1)+"/"+str(len(big_subtrees)))
            last_log_id_1 = id_1
        if id_1 > id_2:
            if set(t_1).isdisjoint(set(t_2)) and max(len(t_1), len(t_2)) > check_threshold_A:
                pair.append((interestingness_index(X, t_1, t_2), (t_1, t_2)))

    if len(pair) == 0:
        main_logger.error("Please consider lowering thresholds. Didn't find apprpriate pair of clusters!")
        exit(1)

    main_logger.info("Considered pairs (disjoint cluster ids) : " + str(len(pair)))
    clusters = [a[1] for a in reversed(sorted(pair))][0]

    # Convert to sets and mark cluster belonging
    clusters = [set(clusters[0]), set(clusters[1])]
    if len(clusters[0]) < len(clusters[1]):
        clusters[0], clusters[1] = clusters[1], clusters[0]

    ids = [i for i in range(K.shape[0]) if i in clusters[0] or i in clusters[1]]
    X, y = _get_raw_data(compound, fingerprint)
    X = X[ids]
    y = y[ids]
    reverse_ids = {i_relative: i_true for i_relative, i_true in enumerate(ids)}
    cluster_id = [1 if reverse_ids[i] in clusters[0] else -1 for reverse_ids[i] in range(y.shape[0])]
    assert len(cluster_id) == y.shape[0]

    if n_folds == 0:
        fold_indices = [[range(y.shape[0]), None]]
    else:
        fold_indices = _generate_fold_indices(cluster_id, n_folds=n_folds, valid_size=valid_size, seed=seed)

    main_logger.info("Creating folds")
    folds = []
    for id, (train_index, valid_index) in enumerate(fold_indices):
        train_index_cluster_0 = [i_relative for i_relative, i_true in enumerate(train_index) if
                                 reverse_ids[i_true] in clusters[0]]
        valid_index_cluster_0 = [i_relative for i_relative, i_true in enumerate(valid_index) if
                                 reverse_ids[i_true] in clusters[0]]
        train_index_cluster_1 = [i_relative for i_relative, i_true in enumerate(train_index) if
                                 reverse_ids[i_true] in clusters[1]]
        valid_index_cluster_1 = [i_relative for i_relative, i_true in enumerate(valid_index) if
                                 reverse_ids[i_true] in clusters[1]]

        # Everything is ok?
        assert(all(reverse_ids[train_index[i]] in clusters[0] for i in train_index_cluster_0))
        assert(all(reverse_ids[train_index[i]] in clusters[1] for i in train_index_cluster_1))



        folds.append({'X_train':{"data":(X[train_index]).copy(),
                                 "cluster_A": train_index_cluster_0,
                                 "cluster_B": train_index_cluster_1,
                                 "i": {"id": id, "raw_size": K.shape[0]}},
                     'Y_train': {"data": (y[train_index]).copy(),
                                 "i": {"id": id}},
                      'X_valid': {"data": (X[valid_index]).copy() if valid_index is not None else np.empty(shape=(0, X.shape[1])),
                                  "cluster_A": valid_index_cluster_0,
                                  "cluster_B": valid_index_cluster_1,
                                  "i": {"id": id, "raw_size": K.shape[0]}
                                   },
                      'Y_valid': { "data": (y[valid_index]).copy() if valid_index is not None else np.empty(shape=(0, )),
                                  "i": {"id": id}}
                      })

    return folds, []

# @cached(save_fnc=joblib_save, load_fnc=joblib_load, check_fnc=joblib_check, cached_ram=True)
def get_splitted_data(compound, fingerprint, n_folds, seed, valid_size, preprocess_fncs=None, test_size=0.0, percent=1):
    """
    Returns data of given compound docked as given fingerprint as folds with training
    and validating data and separate test data

    :param compound desired compound
    :param fingerprint desired fingerprint
    :param n_folds number of folds in train/valid data
    :param test_size test dataset (final validation) is 0.1*100% * number of examples
    """
    X, y = _get_raw_data(compound, fingerprint)
    X = X[:percent * X.shape[0]]
    y = y[:percent * y.shape[0]]
    return  _split(X, y, n_folds=n_folds, seed=seed, test_size=test_size, valid_size=valid_size)



def _split(X, y, n_folds, seed, valid_size, test_size):
    test_data = []
    if test_size > 0:
        split_indices = StratifiedShuffleSplit(y, n_iter=10, test_size=test_size, random_state=seed)
        assert len(split_indices) == 1
        for train_index, test_index in split_indices:
            X_test, y_test = X[test_index], y[test_index]
            X, y = X[train_index], y[train_index]
        test_data = [{"X": {"data":X_test,"i": {"id": 0}},
                    "Y": {"data": y_test, "i": {"id": 0}}
                    }]

    if n_folds == 0:
        fold_indices = [[range(y.shape[0]), None]]
    else:
        # This ensures that proportion of data is maintained
        fold_indices =  _generate_fold_indices(y, n_folds=n_folds, valid_size=valid_size, seed=seed)
        # fold_indices = StratifiedKFold(y, n_folds=n_folds, shuffle=True, random_state=seed)

    folds = []
    for id, (train_index, valid_index) in enumerate(fold_indices):
        folds.append({'X_train':{"data":(X[train_index]).copy(), "i": {"id": id}},
                     'Y_train': {"data": (y[train_index]).copy(), "i": {"id": id}},
                      'X_valid': {"data": (X[valid_index]).copy() if valid_index is not None else np.empty(shape=(0, X.shape[1])),
                                   "i": {"id": id}
                                   },
                      'Y_valid': { "data": (y[valid_index]).copy() if valid_index is not None else np.empty(shape=(0, )),
                                    "i": {"id": id}}
                      })

    return folds, test_data

#### Preprocess functions ####
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
def to_binary(fold, others_to_preprocess=[], threshold_bucket=0, all_below=False):
    """
    @param implicative_ones If bucket i is 1 then i-1...0 are 1
    """
    X_train, Y_train, X_valid, Y_valid = \
        fold["X_train"]["data"].astype("int32"), fold["Y_train"]["data"].astype("int32"), \
        fold["X_valid"]["data"].astype("int32"), fold["Y_valid"]["data"].astype("int32")

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
    fold["X_train"]["data"] = transformer.fit_transform(D)

    if X_valid.shape[0]:
        fold["X_valid"]["data"] = transformer.transform(to_dict_values(X_valid)[0])

    # Wychodzi 0 dla valid and test
    assert(len(others_to_preprocess ) <= 1)
    if len(others_to_preprocess):
        X = others_to_preprocess[0]["X"]["data"]
        Y = others_to_preprocess[0]["Y"]["data"]
        if X.shape[0]:
            D, _ = to_dict_values(X.astype("int32"))
            others_to_preprocess[0]["X"]["data"] = transformer.transform(D)
            others_to_preprocess[0]["Y"]["data"] = Y
        else:
            others_to_preprocess[0]["X"]["data"] = X.astype("int32")
            others_to_preprocess[0]["Y"]["data"] = Y
    return fold, others_to_preprocess


import kaggle_ninja
kaggle_ninja.register("get_splitted_data", get_splitted_data)
kaggle_ninja.register("get_splitted_data_clusterwise", get_splitted_data_clusterwise)
kaggle_ninja.register("get_splitted_data_checkerboard", get_splitted_data_checkerboard)
kaggle_ninja.register("get_splitted_uniform_data", get_splitted_uniform_data)
kaggle_ninja.register("to_binary", to_binary)
