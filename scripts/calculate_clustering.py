#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Calculates clustering using reference clustering
"""

import optparse
import os
import logging
from os import path
import glob
import cPickle

import numpy as np

from misc.utis import to_abs
from misc.config import DATA_DIR
import pandas as pd
from sklearn import cluster
from misc.utils import config_log_to_file
import cPickle
from misc.utils import config_log_to_file
from training_data.datasets import CVBaseChemDataset

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

parser = optparse.OptionParser()
parser.add_option("--clustering_dir", type="str", default="ref_clustering")

if __name__ == "__main__":
    (opts, args) = parser.parse_args()

    clustering_dir = opts.clustering_dir

    compounds = [path.basename(dirname) for dirname in glob.glob(path.join(DATA_DIR, clustering_dir, "*"))]



    config_log_to_file("calculate_clustering.log")
    logger = logging.getLogger("calculate_clustering")

    for compound in compounds:
        fingerprints = [path.splitext(path.basename(f).split("_")[-1])[0] for f \
                        in glob.glob(path.join(DATA_DIR, "ref_clustering", compound, "*"))]
        fingerprints = set(fingerprints)
        assert len(fingerprints) > 0, "Not found any files"

        for fingerprint in fingerprints:

            if fingerprint not in ['PubchemFP']:
                logger.info("Skipping " + fingerprint)
                continue

            logger.info("Processing " + fingerprint + " " + compound)

            cluster_files = glob.glob(path.join(DATA_DIR, clustering_dir, compound, "*{}*".format(fingerprint)))

            clusters = [pd.read_csv(cluster_file, sep=",") for cluster_file in cluster_files]

            cluster_samples = {}
            for cluster, cluster_file in zip(clusters, cluster_files):
                cluster_name = "_".join(cluster_file.split("_")[4:7])
                X_cluster = cluster.values[:,1:].astype("int")
                if cluster_name in cluster_samples:
                    cluster_samples[cluster_name] = np.vstack([cluster_samples[cluster_name], X_cluster])
                else:
                    cluster_samples[cluster_name] = X_cluster

            for cluster_file in cluster_files:
                cluster_name = "_".join(cluster_file.split("_")[4:7])
                N_samples = len(cluster_samples[cluster_name])
                source_files = glob.glob(path.join(DATA_DIR, clustering_dir, compound, "*" + cluster_name + "*{}*".format(fingerprint)))
                N_samples_2 = sum([len(open(f).read().splitlines()) for f in source_files])
                assert N_samples == N_samples_2 - 2, "We should have read all examples"
                assert len(source_files) == 2, "We should have taken actives and inactives into cluster"

            # Merge everything into single matrix for speed of computation
            X_c = np.vstack(list(cluster_samples.values())).astype("float")

            y_c = []
            for cluster_id, cluster_file in enumerate(cluster_samples):
                y_c += [cluster_id for _ in range(cluster_samples[cluster_file].shape[0])]

            y_c = np.array(y_c)

            logger.info("Loading data to be clustered")
            data = CVBaseChemDataset(representation="Pubchem", compound="5-HT1a", n_folds=5)
            # Clipping. MACCS is binary by definition, but Krzysztof MACCS is count
            (X_train, y_train), (X_valid, y_valid) = data.get_data(fold=0)
            X = np.vstack([X_train.todense(), X_valid.todense()])

            d = min(X.shape[1], X_c.shape[1])
            logger.info("Calculating jaccard distances to all cluster samples for train fold")
            K_cluster = _calculate_jaccard_kernel(X[:, 0:d], X_c[:, 0:d])
            assert K_cluster.max() <= 1
            assert K_cluster.min() >= 0
            ids_samples = y_c[np.argmax(K_cluster, axis=1)]

            target_file = os.path.join(DATA_DIR, fingerprint[0:-2], compound + "_" + fingerprint + ".meta")
            logger.info("Dumping clustering information to " + target_file)
            with open(target_file, "w") as f:
                cPickle.dump({"clustering": ids_samples}, f)
