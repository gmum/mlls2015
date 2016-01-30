import sys
from sklearn.utils import check_random_state
import sys
from misc.utils import *
from models.uct import *
from alpy2.utils import masked_indices
from alpy2.strategy import *
from sklearn.metrics import pairwise_distances
from copy import deepcopy
import logging
from itertools import combinations
from cython_routines import score_cython

logger = logging.getLogger(__name__)


class SetFunctionOptimizerGame(object):
    """
    Simple cast of set function optimization as UCT-compatible game

    In cluster case actions are cluster ids and state are selected ids, whereas
    in regular case actions and state are both true ids.

    Parameters
    ----------
    X: list of ids or list of list of ids

    batch_size: int
      How many samples do we want to collect

    scorer:
      Can be called, as scorer.score(ids), and returns objective function over ids.
      In cluster scenario, we should also have scorer.score_cluster_elements(cluster_id)

    element_picking_function: string, default: "max"
      Which element to pick from cluster? Possible values are "max", "prop" and "random".
    """

    def __init__(self, X, scorer, batch_size, rng, element_picking_function="max", use_distance_cache=False):
        assert batch_size > 0, "C'mon!"
        assert element_picking_function in ['max', 'prop', 'random']

        self.rng = check_random_state(rng)
        self.scorer = scorer
        self.use_distance_cache = hasattr(self.scorer, "score_with_cache") and use_distance_cache
        self.X = X
        self.batch_size = batch_size
        self.element_picking_function = element_picking_function

        if isinstance(self.X[0], list):
            self.cluster_ids = set(range(len(self.X)))
            self.cluster_scores = [self.scorer.score_cluster(id) for id in self.cluster_ids]
            self.all_ids = [set(range(len(cluster))) for cluster in self.X]
        else:
            # Assume already all are unknown
            self.all_ids = set(range(len(X)))

    def get_actions(self, state):
        """ Returns list of actions """

        if isinstance(self.X[0], list):
            # Reduce action to just picking cluster
            return list(self.cluster_ids.difference(state['cluster_ids']))
        else:
            return list(self.all_ids.difference(state["ids"]))

    def transition(self, state, action, copy=True):
        """ Transforms state """
        if copy:
            state = deepcopy(state)

        if isinstance(self.X[0], list):
            state['cluster_ids'].append(action)
            state['ids'].append(self._cluster_to_element(action))
        else:
            state["ids"].append(action)

        if self.use_distance_cache:
            state['distance_cache'] = self.scorer.calculate_cache(state['ids'])

        return state

    def utility(self, state):
        """ State -> quasi greedy cost. We want to minimize it. """
        return -self.scorer.score(state['ids'])


    def playout(self, state):
        state = deepcopy(state)

        if isinstance(self.X[0], list):
            left_ids = list(self.cluster_ids.difference(state['cluster_ids']))
            added_cluster_ids = self.rng.choice(left_ids, self.batch_size - len(state['cluster_ids']), replace=False).tolist()
            added_ids = [self._cluster_to_element(cluster_id) for cluster_id in added_cluster_ids]
        else:
            left_ids = list(self.all_ids.difference(state['ids']))
            added_ids = self.rng.choice(left_ids, self.batch_size - len(state['ids']), replace=False).tolist()

        state['ids'] += added_ids
        assert len(state['ids']) == self.batch_size
        return state


    def playout_and_score(self, state):
        if isinstance(self.X[0], list):
            left_ids = list(self.cluster_ids.difference(state['cluster_ids']))
            added_cluster_ids = self.rng.choice(left_ids, self.batch_size - len(state['cluster_ids']), replace=False).tolist()
            added_ids = [self._cluster_to_element(cluster_id) for cluster_id in added_cluster_ids]
            assert len(added_cluster_ids) + len(state['ids']) == self.batch_size
        else:
            left_ids = list(self.all_ids.difference(state['ids']))
            added_ids = self.rng.choice(left_ids, self.batch_size - len(state['ids']), replace=False).tolist()

        if self.use_distance_cache:
            return -self.scorer.score_with_cache(added_ids, state['ids'], state['distance_cache'])
        else:
            return -self.scorer.score(state['ids'] + added_ids)

    def get_key(self, state):
        return tuple(sorted(state['ids']))

    def is_terminal(self, state):
        return len(state["ids"]) == self.batch_size

    def _cluster_to_element(self, cluster_id):
        element_scores = self.scorer.score_cluster(cluster_id)
        assert len(element_scores) == len(self.X[cluster_id])
        if self.element_picking_function == "max":
            return self.X[cluster_id][np.argmax(element_scores)]
        elif self.element_picking_function == "prop":
            return self.rng.choice(len(self.X[cluster_id]), p=element_scores/element_scores.sum())
        elif self.element_picking_function == "random":
            return self.rng.choice(len(self.X[cluster_id]))
        else:
            raise NotImplementedError()


class QuasiGreedyBatchScorer(object):
    """
    Implementation of scorer (to be used in SetFunctionOptimizerGame)

    ----------
    clustering: np.array, shape: (n_samples, )
      Each sample is assigned cluster id
    """

    def __init__(self, X, y, distance_cache, model, base_strategy, batch_size, c, rng, optim=2, clustering=None):
        self.model = model
        self.optim = optim
        self.y = y
        self.c = c
        self.base_strategy = base_strategy
        self.rng = rng
        self.distance_cache = distance_cache
        _, base_scores_masked = base_strategy(X, self.y, rng=np.random.RandomState(self.rng),
                                              model=model, batch_size=batch_size, return_score=True)
        self.base_scores = np.zeros_like(y.data).astype("float32")
        self.base_scores[masked_indices(y)] = base_scores_masked

        # TODO: this check is most likely not needed
        if isinstance(self.base_scores, np.ma.MaskedArray):
            self.base_scores = self.base_scores.data
            logger.warning("Strategy returned base_scores as MaskedArray")

        self.base_scores = self.base_scores.astype("float32")
        self.distance_cache = self.distance_cache.astype("float32")

        self.clustering = clustering

        if self.clustering is not None:
            self._cluster_to_scores = {id: self.base_scores[np.where(clustering == id)[0]] for id in
                                       set(list(clustering))}

    def score_cluster(self, id):
        assert self.clustering is not None, "Requires clustering parameter"
        return self._cluster_to_scores[id]

    def calculate_cache(self, ids):
        if len(ids) < 2:
            return 0

        all_pairs_x, all_pairs_y = zip(*list(combinations(ids, r=2)))
        return self.distance_cache[all_pairs_x, all_pairs_y].sum()

    def score_with_cache(self, ids_new, ids_previous, cache_previous=0):
        N1, N2 = len(ids_new), len(ids_previous)

        if len(ids_new) == 0 or len(ids_previous) == 0:
            all_pairs_x2, all_pairs_y2 = [], []
        else:
            all_pairs_x2, all_pairs_y2 = zip(*(list(product(ids_new, ids_previous))))

        if len(ids_new) < 2:
            all_pairs_x1, all_pairs_y1 = [], []
        else:
            all_pairs_x1, all_pairs_y1 = zip(*(list(combinations(ids_new, r=2))))

        all_pairs = (N1 + N2) * (N1 + N2 - 1 ) / 2.0

        assert len(ids_previous) * (len(ids_previous) - 1)/2.0 + len(all_pairs_y2) + len(all_pairs_x1) == all_pairs, \
            "Covered all pairs"

        return (1. - self.c) / (N1 + N2) * (self.base_scores[ids_new].sum() + self.base_scores[ids_previous].sum())  + \
               (self.c / all_pairs) * (self.distance_cache[all_pairs_x1, all_pairs_y1].sum() + \
                                       self.distance_cache[all_pairs_x2, all_pairs_y2].sum() + cache_previous)

    def score(self, ids, remap=True):
        if remap:
            # Ids are in "unknown" indexing if remap is passed
            ids = masked_indices(self.y)[ids]

        assert len(ids) >= 2, "little ids" + str(ids)  # Otherwise there are no pairs

        if self.optim == 1:
            all_pairs_x, all_pairs_y = zip(*list(combinations(ids, r=2)))# zip(*product(ids, ids))
            # Product has n^2 while correct number is n * (n - 1) / 2.0
            # all_pairs = (len(ids) * (len(ids) - 1))
            # Should have this length
            all_pairs = (len(ids) * (len(ids) - 1)) / 2.0
            assert len(all_pairs_x) == all_pairs
            return (1. - self.c) * self.base_scores[ids].mean() + \
                   (self.c / all_pairs) * self.distance_cache[all_pairs_x, all_pairs_y].sum()
        elif self.optim == 2:
            return score_cython(np.array(ids).astype("int"), self.distance_cache, self.base_scores, self.c)
        else:
            all_pairs_x, all_pairs_y =  zip(*product(ids, ids))
            # Product has n^2 while correct number is n * (n - 1) / 2.0
            all_pairs = (len(ids) * (len(ids) - 1))
            return (1. - self.c) * self.base_scores[ids].mean() + \
                   (self.c / all_pairs) * self.distance_cache[all_pairs_x, all_pairs_y].sum()
