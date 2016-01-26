# -*- coding: utf-8 -*-
"""
 Simple standalone implementation of UCT algorithm
"""

import recordtype
import random
from collections import namedtuple
import math
import numpy as np
import copy
import logging
from sklearn.utils import check_random_state
from copy import deepcopy
from itertools import product
from alpy_addons.strategy import UncertaintySampling, QuasiGreedyBatch
from alpy.utils import masked_indices, unmasked_indices

logger = logging.getLogger(__name__)

UCTNode = recordtype.recordtype("UCTNode", ["id", "state", "children", \
                                            "Q", "N", "parent", "actions", "current_action"])

def printer(self):
    return "UCTNode#{6}: State={0}, Children#={1}, Q={2}, N={3}, actions={4}, current_action={5}".format(
        self.state, len(self.children), self.Q, self.N, self.actions, self.current_action, self.id
    )
UCTNode.__str__ = printer
UCTNode.__repr__ = printer

def eps_greedy_policy(node, c=0.4, d=1):
    E = min(1, (c * len(node.children)) / (d ** 2 * node.N))
    # Probability E
    if random.random() < E:
        if len(node.children) == 0:
            print node
        return np.random.randint(0, len(node.children))
    # Probability 1-E
    else:
        return np.argmax([n.Q / float(n.N) for n in node.children])


def ucb_policy(node, C=np.sqrt(2)):
    if len(node.children) == 0:
        print node
        raise ValueError("Empty children")

    L = [n.Q / float(n.N) + C * math.sqrt(2 * math.log(sum(n2.N for n2 in node.children)) / n.N)
         for n in node.children]
    return np.argmax(L)

class UCT(object):
    """
    Parameters
    ----------
    N: int
      How many iterations to run

    game: UCTGame
      Instance implementing UCTGame interface

    policy: fnc
      Function of signature (node) that selects next action index

    rng: int or np.random.RandomState

    progressive_widening: bool, default: False
      If True will consider node expanded if number_of_actions**0.25 actions have been considered
    """

    def __init__(self, N, game, policy=eps_greedy_policy, rng=None, progressive_widening=False):
        self.policy = policy
        self.game = game
        self.progressive_widening = progressive_widening
        self.N = N
        self.max_id = 0
        self.nodes = None
        self.rng = rng

    def fit(self, state):
        """
        Fits UCT starting from given state
        """
        self._init(state, clean=True)
        self._run(self.N)
        return self

    def partial_fit(self, state, N):
        """
        Fits UCT starting from given state and reusing previous knowledge
        """
        self._init(state, clean=False)
        self._run(N)
        return self

    def _init(self, state, clean=True):
        if self.nodes is None or clean:
            self.nodes = {}

        if clean:
            self.max_id = 0

        if self.game.get_key(state) in self.nodes:
            self.root = self.nodes[self.game.get_key(state)]
        else:
            self.root = UCTNode(id=0, state=state, children=[], Q=0, N=0, \
                                current_action=0, actions=self.game.get_actions(state), parent=None)

        self.rng = check_random_state(self.rng)

    def _run(self, N):
        for i in range(N):
            # Select node
            node = self._tree_policy(self.root)
            if i % int(N / 10.0 + 1) == 0:
                logger.debug("{}/{} iterations processed".format(i, N))

            # Playout and propagate reward
            if self.game.is_terminal(node.state):
                delta = self.game.utility(node.state)
            else:
                delta = self.game.utility(self.game.playout(node.state))
            self._propagate(node, delta)

        # Call policy without exploration
        self.best_action_ = self.root.actions[np.argmax([n.Q / float(n.N) for n in self.root.children])]
        best_path = [self.root]
        node = self.root
        while len(node.children):
            node = node.children[np.argmax([n.Q / float(n.N) for n in node.children])]
            best_path.append(node)
        self.best_path_ = best_path

    def _expand(self, node):
        if self.progressive_widening:
            return len(node.children) < np.floor(node.N ** 0.25) + 1 and len(node.actions) > node.current_action
        else:
            return len(node.actions) != node.current_action

    def _tree_policy(self, node):
        depth = 0
        while not self.game.is_terminal(node.state):
            depth += 1
            if self._expand(node):
                action = node.actions[node.current_action]
                node.current_action += 1

                new_state = self.game.transition(node.state, action)
                existing_state = self.nodes.get(self.game.get_key(new_state), None)
                # Dirty assert for our specific use-case
                # TODO: move it into Game itself if possible
                if existing_state and len(existing_state.state["ids"]) <= len(node.state["ids"]):
                    raise ValueError("Backward edge")

                if existing_state:
                    node.children.append(existing_state)
                    return existing_state
                else:
                    self.max_id += 1
                    new_node = UCTNode(id=self.max_id, state=new_state, parent=node, N=0, Q=0, \
                                       children=[], current_action=0,
                                       actions=self.game.get_actions(new_state))
                    node.children.append(new_node)
                    return new_node
            else:
                node = node.children[self.policy(node)]

            if depth > 100:
                raise ValueError("Most likely there is a cycle in the state graph (reached depth 100)")

        return node

    def _propagate(self, node, delta):
        while node is not None:
            node.N += 1
            # Minus because we are doing argmax always, so if child loses we win
            node.Q += -delta[node.state['player']] if isinstance(delta, list) else -delta
            node = node.parent


    def to_graphviz(self, max_depth=10):
        import graphviz as gv
        g = gv.Graph(format='png')

        def construct_graph_dfs(node, depth=0):
            if depth > max_depth:
                return
            g.node(str(node.id),
                   label=self.game.repr(node.state) + "\n" + str(node.Q / float(node.N)) + ":" + str(node.N))
            for child in node.children:
                g.edge(str(node.id), str(child.id))
                construct_graph_dfs(child, depth + 1)

        construct_graph_dfs(self.root)
        return g



class CrossAndCircle(object):
    """ Simple implementation of Cross and Circle game """

    def __init__(self):
        self.rng = np.random.RandomState(777)

    def is_terminal(self, state):
        return np.all((state["board"] == 0) + (state["board"] == 1)) \
               or sum(self.utility(state)) != 0

    def transition(self, state, action):
        new_state = copy.deepcopy(state)
        new_state["board"][action] = new_state["player"]
        new_state["player"] = (state["player"] + 1) % 2
        return new_state

    def utility(self, state):
        def won(A):
            if any(np.array_equal(r, [1, 1, 1]) for r in A) or \
                    any(np.array_equal(c, [1, 1, 1]) for c in A.T) or \
                    (A[0, 0] == A[1, 1] == A[2, 2] == 1) or \
                    (A[2, 0] == A[1, 1] == A[0, 2] == 1):
                return 1
            else:
                return 0

        rev_board = state["board"].copy()
        rev_board[rev_board == 0] = 2
        rev_board[rev_board == 1] = 0
        rev_board[rev_board == 2] = 1

        if won(rev_board):
            return [1, -1]
        if won(state["board"]):
            return [-1, 1]

        return [0, 0]

    def get_actions(self, state):
        return zip(*np.where(((state['board'] == "x"))))

    def repr(self, state):
        return "\n".join(" ".join([str(v) for v in row]) for row in state["board"])

    def get_key(self, state):
        return str(state['board'])

    def playout(self, state):
        state = copy.deepcopy(state)
        actions = self.get_actions(state)
        self.rng.shuffle(actions)
        player = state["player"]
        for a in actions:
            state["board"][a] = player
            player = (player + 1) % 2
        return state


class UCTStrategyOptimizer(object):
    """
    Simple implementation of optimizer looking for best set of ids for given strategy scorer

    Parameters
    ----------
    y: np.array or list of np.arrays

    batch_size: int
      How many samples do we want to collect?

    Note
    ----
    If y is passed as list of np.arrays it is assumed to represent cluster ids.

    It is advised to shuffle examples first
    """

    def __init__(self, X, y, scorer, batch_size, rng):
        self.y = y
        self.rng = check_random_state(rng)
        self.scorer = scorer
        self.X = X
        self.batch_size = batch_size

        if isinstance(self.y, list):
            assert isinstance(self.X, list)
            self.unknown_ids = [set(masked_indices(cluster)) for cluster in self.y]
            raise NotImplementedError("Calculate self.base_scores here")
        else:
            self.unknown_ids = set(masked_indices(self.y))

    def get_actions(self, state):
        """ Returns list of action, where each action is of format(example_id, cluster_id) """

        if isinstance(self.y, list):
            raise NotImplementedError("Not implemented cluster case")
            return list(enumerate(self.unknown_ids.difference(state["ids"])))
        else:
            return list(self.unknown_ids.difference(state["ids"]))

    def transition(self, state, action, copy=True):
        """ Transforms state """
        if copy:
            state = deepcopy(state)
        state["ids"].append(action)
        if isinstance(self.y, list):
            raise NotImplementedError("Not implemented cluster case")
            state["cluster_ids"].append(action[1])
        return state

    def utility(self, state):
        """ State -> quasi greedy cost. We want to minimize it. """
        return -self.scorer(state['ids'])

    def playout(self, state):
        state = deepcopy(state)
        actions = self.get_actions(state)
        self.rng.shuffle(actions)
        # TODO: change to just picking next samples as a random choice
        # (this is same thing)
        for a in actions:
            self.transition(state, a, copy=False)
            if self.is_terminal(state):
                break

        return state

    def get_key(self, state):
        return str(sorted(state['ids']))

    def is_terminal(self, state):
        return len(state["ids"]) == self.batch_size

    def repr(self, state):
        return str(sorted(list(zip(state["ids"], state.get("cluster_ids", [])))))


class QuasiGreedyBatchScorer(object):
    def __init__(self, X, y, distance_cache, model, base_strategy, batch_size, c, rng):
        self.model = model
        self.c = c
        self.base_strategy = base_strategy
        self.rng = rng
        self.distance_cache = distance_cache
        _, base_scores_masked = base_strategy(X, y, rng=np.random.RandomState(self.rng),
                                       model=model, batch_size=batch_size, return_score=True)
        self.base_scores = np.zeros_like(y).astype("float32")
        self.base_scores[masked_indices(y)] = base_scores_masked
        self.strategy = QuasiGreedyBatch(distance_cache=distance_cache, c=self.c)

    def __call__(self, ids):
        all_pairs_x, all_pairs_y = zip(*product(ids, ids))
        # Product has n^2 while correct number is n * (n - 1) / 2.0
        all_pairs = (len(ids) * (len(ids) - 1))
        return (1.-self.c)*self.base_scores[ids].mean() + \
                (self.c/all_pairs)*self.distance_cache[all_pairs_x, all_pairs_y].sum()