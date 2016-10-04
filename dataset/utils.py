#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilities for datasets and data processing
"""

import scipy.sparse as sp
import numpy as np


def _calculate_jaccard_kernel(X1T, X2T):
    if sp.issparse(X1T) and sp.issparse(X2T):
        X1T_sums = np.array((X1T.multiply(X1T)).sum(axis=1))
        X2T_sums = np.array((X2T.multiply(X2T)).sum(axis=1))
    elif isinstance(X1T, np.ndarray) and isinstance(X2T, np.ndarray):
        X1T_sums = np.array((X1T**2).sum(axis=1))
        X2T_sums = np.array((X2T**2).sum(axis=1))
    else:
        raise NotImplementedError("Not implemented jaccard kernel")

    K = X1T.dot(X2T.T)

    if hasattr(K, "toarray"):
        K = K.toarray()

    K2 = -(K.copy())
    K2 += (X1T_sums.reshape(-1, 1))
    K2 += (X2T_sums.reshape(1, -1))
    K = K / K2

    return K


def _kernel_to_distance_matrix(K):
    if not K.shape[0] == K.shape[1]:
        raise RuntimeError("Pass kernel matrix")

    D = np.sqrt(-2*K + K.diagonal().reshape(-1,1) + K.diagonal().reshape(1,-1))
    D /= D.max()
    return D