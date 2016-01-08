# -*- coding: utf-8 -*-
"""
Simplistic methods for caching function results.

NOTE: Correct usage is adding @cached for prototyping purposes
not final version
"""
from misc.config import CACHE_DIR
from os import path, system
from six import string_types, iteritems
from collections import OrderedDict
from sklearn.base import BaseEstimator
import numpy as np
import cPickle, gzip, pickle
import logging
import inspect

logger = logging.getLogger(__name__)

def get_default_args(func):
    """
    returns a dictionary of arg_name:default_values for the input function
    """
    args, varargs, keywords, defaults = inspect.getargspec(func)
    if defaults:
        return dict(zip(args[-len(defaults):], defaults))
    else:
        return dict()

def cached(func):
    assert len(get_default_args(func)) == 0, "Function cannot take default arguments"

    def cached_func(*args, **kwargs):
        assert len(args) == 0, "No unnamed arguments should be passed."
        cache_kwargs = OrderedDict()
        for key in kwargs:
            assert key not in cache_kwargs, "Passed repeated key (nested BaseEstimator?)."

            if isinstance(kwargs[key], (int, float)) or isinstance(kwargs[key], string_types) or\
                    isinstance(kwargs[key], BaseEstimator):
                cache_kwargs[key] = kwargs[key]
            else:
                raise RuntimeError("Not supported value type for argument.")

        system("mkdir -p " + path.join(CACHE_DIR, func.__module__))
        target = path.join(CACHE_DIR, func.__module__, func.__name__) + "_"
        for key, val in iteritems(cache_kwargs):
            target += key + "=" + str(cache_kwargs[key]).replace(" ", "").replace("\n", "").replace(",", "_") + "_"

        # Try loading
        data = None
        if path.exists(target + ".pkl.gz"):
            try:
                return cPickle.load(gzip.open(target + ".pkl.gz"))
            except Exception, e:
                logger.error("Failed loading cached value with error {}".format(e))
                data = None
        elif path.exists(target + ".npz"):
            try:
                return np.load(open(target + ".npz"))['arr_0']
            except Exception, e:
                logger.error("Failed loading cached value with error {}".format(e))
                data = None

        if data is None:
            logger.info("Caching to " + target)
            data = func(**kwargs)
            if isinstance(data, np.ndarray):
                np.savez(open(target + ".npz", "wb"), data)
            else:
                cPickle.dump(data, gzip.open(target + ".pkl.gz", "wb"))
            return data

    return cached_func

