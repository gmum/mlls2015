from collections import defaultdict
import copy, math, sys
import numpy as np
from collections import namedtuple
import copy
import sys
sys.path.append("..")
from misc.config import *
log = main_logger
import time, random
import unittest
from models.uct import UCT, cross_and_circle, ucb_policy, eps_greedy_policy

class TestUCTStrategies(unittest.TestCase):

    def test_basic_cross_and_circle(self):
        def playout_uct_vs_random(budget=100, policy=ucb_policy):
            global cross_and_circle

            state = {"board": np.array([["x", "x", "x"], ["x", "x", "x"], ["x", "x", "x"]], dtype=object),
                                  "player": 0}

            while not cross_and_circle.is_terminal(state):
                if state['player'] == 0:
                    uct = UCT(N=budget, policy=policy)
                    uct.fit(cross_and_circle, state)
                    state = cross_and_circle.transition(state, uct.best_action)
                else:
                    state = cross_and_circle.transition(state, random.choice(cross_and_circle.get_actions(state)))

            return cross_and_circle.utility(state)[0] == 1

        budget = 60
        wins_0 = 0
        for i in range(50):
            wins_0 += playout_uct_vs_random(budget=budget)
        self.assertTrue(wins_0 > 30)


if __name__ == "__main__":
    unittest.main()