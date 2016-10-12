#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from drgmum.toolkit.dataset import Dataset, SmartDataset
from drgmum.toolkit.dataset.params_to_path import create_params_to_path
from config import CONFIG
from dataset.utils import max_abs
import numpy as np
from sklearn_utils.metrics.pairwise import tanimoto_similarity

import json
import os


class ActiveDataset(object):

    def __init__(self, preprocess=max_abs, warm_start=0.05, **kwargs):
        self.kwargs = kwargs.copy()
        if CONFIG['data_dir'] is None:
            raise RuntimeError("Please call drgmum.config")
        self.dataset = Dataset(
            create_params_to_path(root_dir=CONFIG['data_dir'], ext="pkl.gz")(**kwargs),
            mode=None, file_format="pkl.gz")

        self.json_filename = None

        self.preprocess = preprocess
        self.warm_start = warm_start

    def initialize(self, fold, rng, jaccard):

        # get alpeh cluster ids
        meta = self.load_meta()
        aleph_mask = np.array(meta['validation_cluster'])
        aleph_ids = np.where(aleph_mask == 1)[0]
        splits = meta['splits']

        # get all data
        df = self.load()
        X = df.drop('y', axis=1).values
        y = df['y'].values.to_dense()

        assert isinstance(fold, int)
        assert 0 <= fold < 5  # dr.gmum datasets always have 5-fold splits

        # get train and test ids
        train_ids = np.array(splits[str(fold)]['train'])
        test_ids = np.array(splits[str(fold)]['test'])

        # adjust warm start to be atleast 100 points
        warm_start_n = max(100, int(self.warm_start * train_ids.shape[0]))

        # save train/test split
        self.X_train = X[train_ids]
        self.y_train = y[train_ids]
        self.X_test = X[test_ids]
        self.y_test = y[test_ids]

        aleph_train_ids = np.intersect1d(train_ids, aleph_ids, assume_unique=True)
        aleph_test_ids = np.intersect1d(test_ids, aleph_ids, assume_unique=True)

        # save aleph cluster
        self.X_aleph_train = X[aleph_train_ids]
        self.y_aleph_train = y[aleph_train_ids]

        self.X_aleph_test = X[aleph_test_ids]
        self.y_aleph_test = y[aleph_test_ids]

        # get random warm start
        self.warm_start_ids = np.random.RandomState(rng).choice(np.where(aleph_mask == 0)[0],
                                                                size=warm_start_n,
                                                                replace=False)

        if jaccard:
            self.X_train = tanimoto_similarity(self.X_train, self.X_train)
            self.X_test = tanimoto_similarity(self.X_test, self.X_train)


    def get_params(self):
        return self.kwargs

    def exists(self):
        return self.dataset.exists()

    def load(self):
        return self.dataset.read()

    def save(self, obj, overwrite=True):
        if not overwrite and self.exists():
            pass

        self.dataset.write(obj)

    def save_meta(self, meta):

        assert isinstance(meta, dict)

        filename = self._get_meta_filename()
        json.dump(meta, open(filename, 'w'))

    def load_meta(self):

        filename = self._get_meta_filename()
        assert os.path.exists(filename)

        return json.load(open(filename, 'r'))

    def _get_meta_filename(self):

        if self.json_filename is not None:
            return self.json_filename
        else:
            self.json_filename = self.dataset.filename[:-6] + "json"
            return self.json_filename





























