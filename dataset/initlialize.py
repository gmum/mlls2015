# -*- coding: utf-8 -*-

"""
Copied from Chembl_Baselines basic example notebook

Downloads all relevant datasets
"""

from StringIO import StringIO
import pandas as pd
import urllib2
import json
from sklearn.datasets import load_svmlight_file
from config import PROTEINS, FINGERPRINTS
from itertools import product
from drgmum.toolkit.dataset import SmartDataset
import logging
import traceback


def download_dataset(which, version, organism, protein):
    def _download(filename):
        url = "http://gmum.net/files/datasets/" + '/'.join([which, version, organism, protein, filename])
        try:
            response = urllib2.urlopen(url)
        except urllib2.HTTPError as e:
            logger.error("Got HTTP Error trying to reach {}".format(url))
            raise e

        return response.read()

    X = pd.read_csv(StringIO(_download("X.csv")), index_col=0, na_values=[], keep_default_na=False)
    y = pd.read_csv(StringIO(_download("y.csv")), index_col=0, na_values=[], keep_default_na=False)
    labels = pd.read_csv(StringIO(_download("labels.csv")), index_col=0, na_values=[], keep_default_na=False)
    meta = pd.read_csv(StringIO(_download("meta.csv")), index_col=0, na_values=[], keep_default_na=False)
    splits = json.loads(_download("splits.json"))
    return X, y, labels, meta, splits


def download_smiles():
    content = urllib2.urlopen("http://gmum.net/files/datasets/features/smiles.csv").read()
    return pd.Series.from_csv(StringIO(content), index_col=0)


def download_fingerprint(which):
    content = urllib2.urlopen("http://gmum.net/files/datasets/features/" + which + "_v1.libsvm").read()
    return load_svmlight_file(StringIO(content))[0]


# helper function used to get the correct rows of Fingerprint

def get_idx(uids, all_uids):
    idx = []
    for uid in uids:
        idx.append(all_uids.index(uid))
    return idx

logger = logging.getLogger(__file__)

for (protein, fingerprint) in product(PROTEINS[:1], FINGERPRINTS[:1]):

    X, y, labels, meta, splits = download_dataset(which="log_Ki_basic",
                                                  version="v1",
                                                  organism="Human",
                                                  protein=protein)

    smiles = download_smiles()
    fp_data = download_fingerprint(fingerprint)

    strat_splits = {}

    for fold in [str(i) for i in xrange(5)]:

        train_uids = splits['StratifiedKFold_2.5'][fold]['train']
        test_uids = splits['StratifiedKFold_2.5'][fold]['test']

        strat_splits[fold] = {'train': get_idx(train_uids, list(smiles.index.values)),
                              'test': get_idx(test_uids, list(smiles.index.values))}


    sd = Smagit add notrtDataset(name="Features", which="Fingerprint", fingerprint=fingerprint, version='v1')
    binary_y = y["log_Ki_thr:(2.5,)"]

    dataset = {'X': fp_data, 'y': binary_y, 'splits': strat_splits}

    if not sd.exists():
        logger.info("Calculating {0} fingerprint for {1}...".format(fingerprint, protein))
        sd.save(dataset)
        logger.info("Done.")
























