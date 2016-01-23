from .utils import COMPOUNDS, FINGERPRINTS, \
    get_libsvm_chemical_file, get_smiles_chemical_files, get_raw_scmaccs_files
from misc.config import *
import os
import numpy as np
import scipy
from os import path
from sklearn.datasets import load_svmlight_file, load_svmlight_files
import cPickle


def load_raw_chemical_data(compound, representation, n_features=None):
    """
    Loads data from a single file given by combination of compound and fingerprint
    """
    if compound not in COMPOUNDS:
        raise ValueError("Bad compound %s" % compound)
    # if representation not in (FINGERPRINTS):
    #     raise ValueError("Bad representation %s" % representation)


    if representation == "smiles":
        assert n_features is None, "n_features is not supported for smiles"
        file_name_actives, file_name_inactives = get_smiles_chemical_files(compound)
        x, y = [], []
        with open(file_name_actives) as f:
            x += f.read().splitlines()
            y += len(x) * [1]

        with open(file_name_inactives) as f:
            x += f.read().splitlines()
            y += (len(x) - len(y)) * [-1]

        return np.array(x, dtype="O").reshape(-1, 1), np.array(y, dtype="O").reshape(-1) # Object type is more memory efficient
    elif representation in FINGERPRINTS:
        file_name = get_libsvm_chemical_file(compound, representation)
        if not os.path.exists(file_name):
            raise ValueError("No data file %s for %s compound and  %s fingerprint" %
                             (file_name, compound, representation))

        return load_svmlight_file(file_name, zero_based=True, n_features=n_features)
    else:
        raise RuntimeError("Not recognized representation")


def load_meta(compound, representation):
    assert representation in FINGERPRINTS
    file_name = get_libsvm_chemical_file(compound, representation)
    base_name = path.splitext(file_name)[0]
    if not os.path.exists(base_name + ".meta"):
        raise RuntimeError("Not found meta file")
    with open(base_name + ".meta", "r") as f:
        data = cPickle.load(f)
    return data
