#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilities for datasets and data processing
"""

import scipy.sparse as sp
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
import scipy


def identity(X_train, y_train, X_valid, y_valid):
    return (X_train, y_train), (X_valid, y_valid)

def max_abs(X_train, y_train, X_valid, y_valid):

    # Converts [0, 4] --> [0, 4/max(feature)]
    norm = MaxAbsScaler()
    X_train = norm.fit_transform(X_train)
    if X_valid.shape[0]:
        X_valid = norm.transform(X_valid)
    return (X_train, y_train), (X_valid, y_valid)


def clip01(X_train, y_train, X_valid, y_valid):

    if isinstance(X_train, scipy.sparse.csr_matrix):
        X_train.data = np.clip(X_train.data, 0, 1)
        X_valid.data = np.clip(X_valid.data, 0, 1)
    else:
        X_train = np.clip(X_train, 0, 1)
        X_valid = np.clip(X_valid, 0, 1)
    return (X_train, y_train), (X_valid, y_valid)