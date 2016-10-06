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
from sklearn_utils.metrics.pairwise import tanimoto_similarity
from dataset import ActiveDataset
import logging
import numpy as np
from clustering import get_chemical_clustering_groups

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

for protein in PROTEINS:

    logger.info(" Generating data for {}".format(protein))

    molprint_data = download_fingerprint("MolPrint2D")

    for fingerprint in FINGERPRINTS:

        logger.info("\t fingerpint {}".format(fingerprint))

        sd = ActiveDataset(name="Features", which="Fingerprint", fingerprint=fingerprint, protein=protein, version='v1')

        if sd.exists():
            logger.info(" Data file already exists, skipping")
            continue

        # download data
        X_all, y_all, labels, meta_all, splits_all = download_dataset(which="log_Ki_basic",
                                                                  version="v1",
                                                                  organism="Human",
                                                                  protein=protein)

        smiles = download_smiles()
        fp_data = download_fingerprint(fingerprint)

        # get splits
        splits = {}

        for fold in [str(i) for i in xrange(5)]:

            train_uids = splits_all['StratifiedKFold_2.5'][fold]['train']
            test_uids = splits_all['StratifiedKFold_2.5'][fold]['test']

            splits[fold] = {'train': get_idx(train_uids, list(smiles.index.values)),
                            'test': get_idx(test_uids, list(smiles.index.values))}

        y = y_all["log_Ki_thr:(2.5,)"].values
        X = fp_data[get_idx(y_all.index, list(smiles.index.values)), :].tocsc()

        ### Validation Cluster

        # use MolPrint2D for clustering
        X_molprint = molprint_data[get_idx(y_all.index, list(smiles.index.values)), :]

        X_kernel = tanimoto_similarity(X_molprint, X_molprint)
        n_clusters = 5

        cluster_ids = get_chemical_clustering_groups(kernel=X_kernel, n_clusters=n_clusters)
        frequencies = [(cluster_ids == id).sum() for id in xrange(n_clusters)]

        # sizes of clusters are not checked, as the clustering used is balanced
        candidates = range(n_clusters)

        # calculate intercluster distances for each point
        min_distances = []
        for cluster in candidates:
            K = 1 - tanimoto_similarity(X_molprint[cluster_ids == cluster], X_molprint[cluster_ids != cluster])
            K = np.min(K, axis=1)
            min_distances.append(np.array(K).reshape(-1))

        # find a intercluster distance value at 5% of data (1st 20-quantile) for checking which cluster has its
        # closest-to-other-clusters points the farthest
        very_close_threshold = np.percentile(np.hstack(min_distances), 5)

        # how many points in each cluster are below the 5% all intercluster distance
        probability_finding_very_close = [sum(x <= very_close_threshold) / float(x.shape[0]) for x in min_distances]

        # pick the best cluster - that has the least % of points below the treshold
        best_candidate_idx = np.argmin(probability_finding_very_close)
        best_candidate = candidates[best_candidate_idx]

        active_percentage_all = sum(y) / float(y.shape[0])
        active_percentage = [sum(y[np.where(cluster_ids == c)]) / float(frequencies[c]) for c in candidates]

        if active_percentage[best_candidate_idx] < 0.5:
            logger.warning("Low active percenatge in best pick: {}".format(active_percentage[best_candidate_idx]))

        meta = {'splits': splits,
                'clustering': cluster_ids.tolist(),
                'validation_cluster': (cluster_ids == best_candidate).astype("int").tolist()}

        # sava data and meta as ActiveDataset

        sdf = pd.SparseDataFrame([pd.SparseSeries(X[i].toarray().ravel())
                                  for i in np.arange(X.shape[0])], index=y_all.index)

        sdf.insert(len(sdf.columns), "y", y)

        sd.save(sdf)
        sd.save_meta(meta)

logger.info("Done.")
























