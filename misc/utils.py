# -*- coding: utf-8 -*-
"""
Simple utils functions
"""

import logging
from config import *
import os
from os import getpid
import numpy as np
from os import path
import datetime
import itertools
import time
import logging
import inspect
from socket import gethostname
from subprocess import Popen, PIPE

def config_log_to_file(fname="mol2vec.log", level=logging.INFO, clear_log_file=False):
    """
    Manual setup for logging to file
    """
    if not path.isabs(fname):
        fname = path.join(LOG_DIR, fname)

    if clear_log_file:
        with open(fname, "w") as f:
            pass

    logger = logging.getLogger('')
    logger.setLevel(level)
    logger.handlers = []
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler(fname)
    fh.setFormatter(formatter)
    fh.setLevel(level)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def utc_date(format="%Y_%m_%d"):
    return datetime.datetime.utcnow().strftime(format)

def utc_timestamp():
    return str(int((datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds()))

def reporting(iterable, K, N=None, log=None):
    for id, it in enumerate(iterable):
        if id % K == 0:
            if N:
                if log:
                    log.info("{0} {1} %".format(id, 100*id/float(N)))
                else:
                    print id, " ", id/float(N), "%"
            else:
                if log:
                    log.info(id)
                else:
                    print id
        yield it




def to_abs(file_name, base_dir=RESULTS_DIR):
    return file_name if path.isabs(file_name) else path.join(base_dir, file_name)

def batched(iterable, size):
    sourceiter = iter(iterable)
    while True:
        batchiter = itertools.islice(sourceiter, size)
        yield itertools.chain([batchiter.next()], batchiter)

def get_run_properties():
    """ Returns a dictionary of relevant properties of the code/environment with which it was run. """
    props = dict()
    # python 2.6 compatible implementation
    # props["git_commit"] = Popen(['git', 'rev-parse', 'HEAD'], stdout=PIPE).communicate()[0]
    frame, filename, line_number, function_name, lines, index = inspect.stack()[1]
    props['filename'] = filename
    if os.path.exists(filename):
        props['code'] = open(filename).read().splitlines()
    props["hostname"] = gethostname()
    props["numpy_version"] = np.__version__
    props["PID"] = getpid()
    return props