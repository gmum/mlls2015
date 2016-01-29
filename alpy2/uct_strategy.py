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

    def __init__(self, X, scorer, batch_size, rng, element_picking_function="max"):
        assert batch_size > 0, "C'mon!"
        assert element_picking_function in ['max', 'prop', 'random']

        self.rng = check_random_state(rng)
        self.scorer = scorer
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
            return list(self.cluster_ids.difference(state['ids']))
        else:
            return list(self.all_ids.difference(state["ids"]))

    def transition(self, state, action, copy=True):
        """ Transforms state """
        if copy:
            state = deepcopy(state)

        if isinstance(self.X[0], list):
            state['ids'].append(self._cluster_to_element(action))
        else:
            state["ids"].append(action)

        return state

    def utility(self, state):
        """ State -> quasi greedy cost. We want to minimize it. """
        return -self.scorer.score(state['ids'])

    def playout(self, state):
        state = deepcopy(state)

        if isinstance(self.X[0], list):
            left_ids = list(self.cluster_ids.difference(state['ids']))
            added_cluster_ids = self.rng.choice(left_ids, self.batch_size - len(state['ids']), replace=False).tolist()
            added_ids = [self._cluster_to_element(cluster_id) for cluster_id in added_cluster_ids]
        else:
            left_ids = list(self.all_ids.difference(state['ids']))
            added_ids = self.rng.choice(left_ids, self.batch_size - len(state['ids']), replace=False).tolist()

        state['ids'] += added_ids
        assert len(state['ids']) == self.batch_size
        return state

    def get_key(self, state):
        return str(sorted(state['ids']))

    def is_terminal(self, state):
        return len(state["ids"]) == self.batch_size

    def _cluster_to_element(self, cluster_id):
        element_scores = self.scorer.score_cluster(cluster_id)
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

    def __init__(self, X, y, distance_cache, model, base_strategy, batch_size, c, rng, clustering=None):
        self.model = model
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

        self.clustering = clustering

        if self.clustering is not None:
            self._cluster_to_scores = {id: self.base_scores[np.where(clustering == id)[0]] for id in
                                       set(list(clustering))}

    def score_cluster(self, id):
        assert self.clustering is not None, "Requires clustering parameter"
        return self._cluster_to_scores[id]

    def score(self, ids, remap=True):
        if remap:
            # Ids are in "unknown" indexing if remap is passed
            ids = masked_indices(self.y)[ids]

        assert len(ids) >= 2, "little ids" + str(ids)  # Otherwise there are no pairs

        all_pairs_x, all_pairs_y = zip(*product(ids, ids))
        # Product has n^2 while correct number is n * (n - 1) / 2.0
        all_pairs = (len(ids) * (len(ids) - 1))
        return (1. - self.c) * self.base_scores[ids].mean() + \
               (self.c / all_pairs) * self.distance_cache[all_pairs_x, all_pairs_y].sum()
