import os
import itertools
import cPickle
import logging
import numpy as np

from sklearn.cross_validation import train_test_split
from misc.config import DATA_DIR
from sklearn.utils import check_random_state
from sklearn.cross_validation import StratifiedKFold

FINGERPRINTS = ["MACCS", "SRMACCS", "SRKMACCS","Klek", "Pubchem", "Ext"]

COMPOUNDS = ['mGluR3', '5-HT2a', '5-HT1b', 'SERT', 'H1', 'cdk2', 'src',
             'beta2', '5-HT2a', 'gly', '5-HT1a', 'lck', 'mGluR3', 'beta2',
             'M2', '5-HT1b', 'abl', '5-HT7', '5-HT6', '5-HT6', 'H1', 'abl',
             'cdk2', 'M2', 'mGluR8', 'gly', '5-HT1a', 'SERT', 'src', 'mGluR8',
             'lck', '5-HT7']

COMPOUNDS = list(set(COMPOUNDS)) # That was lazy

COMPOUNDS += [c + "_DUDs" for c in COMPOUNDS]

COMPOUNDS_BIGGER = ['H1', 'abl', '5-HT2a', 'lck', '5-HT1a', 'SERT', 'src']

COMPOUNDS_SIZES = {'5-HT1a': 1155,
                   '5-HT1b': 297,
                   '5-HT2a': 976,
                   '5-HT2b': 333,
                   '5-HT6': 341,
                   '5-HT7': 370,
                   'H1': 558,
                   'M2': 261,
                   'SERT': 1596,
                   'abl': 597,
                   'beta2': 275,
                   'cdk2': 109,
                   'gly': 51,
                   'lck': 1136,
                   'mGluR3': 48,
                   'mGluR8': 2,
                   'src': 1702}



def shuffle_in_sync(arrs, rng):
    """ Shuffles data in x, y using rng, ensures that the rows of x and y
    mantain their alignment. """
    rng_state = rng.get_state()
    for arr in arrs:
        rng.shuffle(arr)
        rng.set_state(rng_state)


def get_libsvm_chemical_file(compound, fingerprint):
    if fingerprint in FINGERPRINTS and fingerprint.startswith("SR"):
        return os.path.join(DATA_DIR, "srfp", fingerprint[2:], compound + "_" +
                            fingerprint[2:] + "_count_reducedcoord.libsvm")
    elif fingerprint in FINGERPRINTS:
        # TODO: remove "FP" suffix
        return os.path.join(DATA_DIR, fingerprint, compound + "_" + fingerprint + "FP.libsvm")
    else:
        raise RuntimeError("Not recognized fingerprint")


def check_binary(x):
    return (np.unique(x.toarray()) == [0, 1]).all()



def _split_ids(x, y, n_folds, rng, fold):
    ids = list(StratifiedKFold(y, n_folds=n_folds, random_state=rng, shuffle=True))[fold]
    np.random.RandomState(rng).shuffle(ids[0])
    np.random.RandomState(rng).shuffle(ids[1])
    return ids[0], ids[1]

def split_data_folds(x, y, n_folds, rng=None, fold=0):
    ids_train, ids_valid = _split_ids(x, y, n_folds, rng, fold)
    x_train, x_valid, y_train, y_valid = x[ids_train], x[ids_valid], y[ids_train], y[ids_valid]
    return (x_train, y_train), (x_valid, y_valid), (ids_train, ids_valid)


def update_meta(fname, meta):
    """ Replaces given key in dict or creates file if didn't exist """
    try:
        old_meta = cPickle.load(open(fname, "r"))
    except Exception, e:
        old_meta = {}
    old_meta.update(meta)
    with open(fname, "w") as f:
        cPickle.dump(old_meta, f)



def split_data(x, y, rng=None, test_size=0.08):
    assert False, "train_test_split most likely is not shuffling ids"
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, random_state=rng, test_size=test_size, stratify=y)
    return (x_train, y_train), (x_valid, y_valid)


def smoke_test():
    # check data files
    required_files = itertools.product(COMPOUNDS, FINGERPRINTS)

    for comp, fp in required_files:
        file_name = get_libsvm_chemical_file(comp, fp)
        os.path.exists(os.path.join(DATA_DIR, file_name))


def get_smiles_chemical_files(compound):
    return [os.path.join(DATA_DIR, "smiles", compound + "_actives.smi"), \
            os.path.join(DATA_DIR, "smiles", compound + "_inactives.smi")]


def get_raw_scmaccs_files(compound):
    return [os.path.join(DATA_DIR, "srfp", "MACCS", compound + "_actives_MACCS_count_coord.dat"), \
            os.path.join(DATA_DIR, "srfp", "MACCS", compound + "_inactives_MACCS_count_coord.dat")]


smoke_test()
