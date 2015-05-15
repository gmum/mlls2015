"""
This file contains utility functions
"""

import time
import os
import cPickle
import numpy
import pickle
import numpy as np
import glob
import mmap
import pandas as pd
from scipy import sparse
import matplotlib
import os

import joblib


def joblib_load(key, cache_dir):
    file_name = os.path.join(cache_dir, key)
    return joblib.load(file_name)

def joblib_check(key, cache_dir):
    return len(glob.glob(os.path.join(cache_dir, key + "*.npy"))) > 0

def joblib_save(key, val, cache_dir):
    file_name = os.path.join(cache_dir, key)
    joblib.dump(val, file_name)

    
def scikit_load(key, cache_dir):
    dir = os.path.join(cache_dir, key)
    file_name = os.path.join(os.path.join(cache_dir, dir), key + ".pkl")
    return joblib.load(file_name)


def scikit_check(key, cache_dir):
    dir = os.path.join(cache_dir, key)
    return len(glob.glob(os.path.join(os.path.join(cache_dir, dir), key + ".pkl*"))) > 0


def scikit_save(key, val, cache_dir):
    dir = os.path.join(cache_dir, key)
    os.system("mkdir " + dir)
    file_name = os.path.join(dir, key + ".pkl")
    joblib.dump(val, file_name)


def scipy_csr_load(key, cache_dir):
    file_name = os.path.join(cache_dir, key + ".npz")
    f = np.load(file_name)
    return sparse.csr_matrix((f["arr_0"], f["arr_1"], f["arr_2"]), shape=f["arr_3"])


def scipy_csr_check(key, cache_dir):
    return os.path.exists(os.path.join(cache_dir, key + ".npz"))


def scipy_csr_save(key, val, cache_dir):
    file_name = os.path.join(cache_dir, key)
    np.savez(file_name, val.data, val.indices, val.indptr, val.shape)


def pandas_save_fnc(key, val, cache_dir):
    file_name = os.path.join(cache_dir, key + ".msg")
    val.to_msgpack(file_name)


def pandas_check_fnc(key, cache_dir):
    return os.path.exists(os.path.join(cache_dir, + key + ".msg"))


def pandas_load_fnc(key, cache_dir):
    file_name = os.path.join(cache_dir, key + ".msg")
    return pd.read_msgpack(file_name)


def numpy_save_fnc(key, val, cache_dir):
    if isinstance(val, tuple):
        raise "Please use list to make numpy_save_fnc work"
    # Note - not using savez because it is reportedly slow.
    if isinstance(val, list):
        logger.info("Saving as list")
        save_path = os.path.join(c["CACHE_DIR"], key)
        save_dict = {}
        for id, ar in enumerate(val):
            save_dict[str(id)] = ar
        np.savez(save_path, **save_dict)
    else:
        logger.info("Saving as array " + str(val.shape))
        np.save(os.path.join(c["CACHE_DIR"], key + ".npy"), val)


def numpy_check_fnc(key, cache_dir):
    return len(glob.glob(os.path.join(cache_dir, key + ".np*"))) > 0


def numpy_load_fnc(key, cache_dir):
    if os.path.exists(os.path.join(cache_dir, key + ".npz")):
        # Listed numpy array

        savez_file = np.load(os.path.join(cache_dir, key + ".npz"))

        ar = []

        for k in sorted(list((int(x) for x in savez_file))):
            logger.info("Loading " + str(k) + " from " + str(key) + " " + str(savez_file[str(k)].shape))
            ar.append(savez_file[str(k)])
        return ar
    else:
        return np.load(os.path.join(cache_dir, key + ".npy"))