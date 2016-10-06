#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Copied from chembl_baselines
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score


def _dist_two_means(scalar_prod, idx1, idx2):
    assert scalar_prod[0,0] > 0, "scalar product, not distance!"
    return scalar_prod[idx1, :][:, idx1].mean() + scalar_prod[idx2, :][:, idx2].mean() - 2 * scalar_prod[idx1, :][:, idx2].mean()

def _dist_clusters(scalar_prod, clustered, mode="mean"):
    shape = (clustered.max() + 1, clustered.max() + 1)
    cl_dist = np.zeros(shape)
    for i in xrange(shape[0]):
        for j in xrange(shape[1]):
            cl_dist[i, j] = _dist_two_means(scalar_prod, np.where(clustered == i)[0], np.where(clustered == j)[0])
    return cl_dist

def _kernel_to_distance(kernel):
    # returns SQUARED norm
    return np.diagonal(kernel).reshape(-1, 1) + np.diagonal(kernel).reshape(1, -1) - 2 * kernel

def _reduce_kernel(kernel, groups):
    u_groups, idx, counts = np.unique(groups, return_inverse=True, return_counts=True)
    reduced_kernel = np.zeros((len(u_groups), len(u_groups)))
    for i in xrange(reduced_kernel.shape[0]):
        for j in xrange(i + 1):
            v = kernel[idx==i,:][:,idx==j].mean()
            reduced_kernel[i, j] = v
            reduced_kernel[j, i] = v
    return reduced_kernel, u_groups, counts

def _reduce_kernel_2(kernel, groups):
    u_groups, idx, counts = np.unique(groups, return_inverse=True, return_counts=True)
    reduced_kernel = np.zeros((len(u_groups), len(u_groups)))
    reduced_denom = np.zeros((len(u_groups), len(u_groups)))
    for i in xrange(len(idx)):
        for j in xrange(len(idx)):
            reduced_kernel[idx[i], idx[j]] += kernel[i, j]
            reduced_denom[idx[i], idx[j]] += 1
    return np.divide(reduced_kernel, reduced_denom), u_groups, counts

class GreedyTimeClustering(BaseEstimator):

    def __init__(self, D=1.43):
        # 1.43 because 43 is my favourite number
        # works fine with serotonin receptors
        self.D = D

    def fit(self, X, y):
        # X - must be precomputed dist
        # y - time labels
        time_idx = np.argsort(y)
        rev_time_idx = np.argsort(time_idx) # this is beautiful...
        X = X[time_idx, :][:, time_idx]
        clusters = {0:[0]}
        next_cluster = 1
        for i in xrange(1, X.shape[0]):
            _l = []
            for n_cluster, indices in clusters.iteritems():
                if X[i, indices].max() <= self.D:
                    _l.append((X[i, indices].min(), n_cluster))
            if len(_l) == 0:
                clusters[next_cluster] = [i]
                next_cluster += 1
            else:
                clusters[sorted(_l)[0][1]].append(i)
        self.labels_ = np.zeros(X.shape[0], dtype=np.int64)
        for n_cluster, indices in clusters.iteritems():
            for i in indices:
                self.labels_[i] = n_cluster
        self.labels_ = self.labels_[rev_time_idx]
        return self

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.labels_

class BalancedAgglomerativeClustering(BaseEstimator):

    class _Tree(object):
        def __init__(self, children, leaf_weights):
            self._INTNAN = -1 # be careful! -1 is a proper index
            self.children = np.array(children)[:]
            self.n_samples = len(leaf_weights)
            self.n_nodes = 2 * self.n_samples - 1
            self.parents = np.empty(self.n_nodes, dtype=np.int64)
            self.parents[:] = self._INTNAN
            for i in xrange(self.children.shape[0]):
                for j in [0, 1]:
                    self.parents[self.children[i, j]] = i + self.n_samples
            self.leaf_weights = np.array(leaf_weights)[:]
        def get_clusters(self, n_clusters):
            leaf_weights = self.leaf_weights[:]
            clusters = []
            used = set()
            while n_clusters > 0:
                leafs = self.find_cluster(float(leaf_weights.sum()) / float(n_clusters), leaf_weights)
                cluster = []
                for leaf in leafs:
                    if not leaf in used:
                        used.add(leaf)
                        cluster.append(leaf)
                        leaf_weights[leaf] = 0
                clusters.append(cluster)
                n_clusters -= 1
            return clusters
        def find_cluster(self, cluster_size, leaf_weights):
            sizes = self.calculate_cluster_sizes(leaf_weights)
            node = np.abs(sizes - cluster_size).argmin()
            # if some leafs have zero weight , choose highest possible node
            # if node has no siblings, go up
            while self.parents[node] != self._INTNAN and sizes[self.parents[node]] == sizes[node]:
                node = self.parents[node]
            return sorted(self.get_leafs(node))
        def calculate_cluster_sizes(self, leaf_weights):
            sizes = np.zeros(self.n_nodes)
            for i in xrange(self.n_samples):
                w = leaf_weights[i]
                current = i
                sizes[current] += w
                while self.parents[current] != self._INTNAN:
                    current = self.parents[current]
                    sizes[current] += w
            return sizes
        def get_leafs(self, node):
            leafs = []
            stack = []
            visited = set()
            stack.append(node)
            while len(stack) > 0:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                if current < self.n_samples:
                    leafs.append(current)
                else:
                    stack.append(self.children[current - self.n_samples, 0])
                    stack.append(self.children[current - self.n_samples, 1])
            return leafs

    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        assert not (np.diagonal(X) == 0).all(), "'X' must be kernel matrix, not distance!"
        kernel = X
        groups = y
        # 'average' seems to be more stable
        #linkage = "complete"
        linkage = "average"
        if groups is None:
            groups = range(kernel.shape[0])
        reduced_kernel, u_groups, counts = _reduce_kernel_2(kernel, groups)
        dist = _kernel_to_distance(reduced_kernel)
        ac = AgglomerativeClustering(affinity="precomputed", linkage=linkage).fit(dist)
        tree = BalancedAgglomerativeClustering._Tree(ac.children_, counts)
        clusters = tree.get_clusters(self.n_clusters)
        new_groups = np.array(groups)
        for i, cluster in enumerate(clusters):
            for ind in cluster:
                new_groups[groups==u_groups[ind]] = i
        self.labels_ = new_groups
        return self

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_

def get_chemical_clustering_groups(kernel, method="bac", n_clusters=5, labels=None):
    dist = _kernel_to_distance(kernel)
    if method == "bac":
        # labels - we expect None
        groups = BalancedAgglomerativeClustering(n_clusters=n_clusters).fit_predict(kernel)
    elif method == "bac_gtc":
        # labels - we expect timestamp
        groups_gtc = GreedyTimeClustering(D=1.43).fit_predict(dist, labels)
        groups = BalancedAgglomerativeClustering(n_clusters=n_clusters).fit_predict(kernel, groups_gtc)
    elif method == "bac_murcko":
        # labels - we expect scaffolds (might be str, but other hashable should be fine)
        groups = BalancedAgglomerativeClustering(n_clusters=n_clusters).fit_predict(kernel, labels)
    elif method == "bac_murcko_generic":
        # labels - we expect scaffolds (might be str, but other hashable should be fine)
        groups = BalancedAgglomerativeClustering(n_clusters=n_clusters).fit_predict(kernel, labels)
    else:
        raise NotImplementedError()
    return groups