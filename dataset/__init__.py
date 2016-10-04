#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from drgmum.toolkit.dataset import Dataset, SmartDataset
from drgmum.toolkit.dataset.params_to_path import create_params_to_path
from .. import CONFIG

class ActiveDataset(object):

    def __init__(self, **kwargs):
        self.kwargs = kwargs.copy()
        if CONFIG['data_dir'] is None:
            raise RuntimeError("Please call drgmum.config")
        self.dataset = Dataset(
            create_params_to_path(root_dir=CONFIG['data_dir'], ext="pkl.gz")(**kwargs),
            mode=None, file_format="pkl.gz")

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