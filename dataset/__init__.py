#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from drgmum.toolkit.dataset import Dataset, SmartDataset
from drgmum.toolkit.dataset.params_to_path import create_params_to_path
from config import CONFIG


import json
import os


class ActiveDataset(object):

    def __init__(self, **kwargs):
        self.kwargs = kwargs.copy()
        if CONFIG['data_dir'] is None:
            raise RuntimeError("Please call drgmum.config")
        self.dataset = Dataset(
            create_params_to_path(root_dir=CONFIG['data_dir'], ext="pkl.gz")(**kwargs),
            mode=None, file_format="pkl.gz")

        self.json_filename = None

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

        return json.load(filename)

    def _get_meta_filename(self):

        if self.json_filename is not None:
            return self.json_filename
        else:
            self.json_filename = self.dataset.filename[:-6] + "json"
            return self.json_filename
























