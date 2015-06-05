import recordtype
import random
from collections import namedtuple
import math
import numpy as np
import copy
__author__ = 'kudkudak'

UCTNode = recordtype.recordtype("UCTNode", ["id", "state", "children", \
                              "Q", "N", "parent", "actions", "current_action"])
def printer(self):
    return "UCTNode#{6}: State={0}, Children#={1}, Q={2}, N={3}, actions={4}, current_action={5}".format(
        self.state, len(self.children), self.Q, self.N, self.actions, self.current_action, self.id
        )
UCTNode.__str__ = printer
UCTNode.__repr__ = printer
UCTGame = namedtuple("UCTGame", ["transition", "repr", "state_key", "playout_randomly", \
                              "get_actions", "is_terminal", "utility"])


def eps_greedy_policy(node, c = 0.4, d = 1):
    E = min(1, (c * len(node.children))/(d**2 * node.N))
    # Probability E
    if random.random() < E:
        if len(node.children) == 0:
            print node
        return np.random.randint(0, len(node.children))
    # Probability 1-E
    else:
        return np.argmax([n.Q/float(n.N) for n in node.children])

def ucb_policy(node, c=0, C = np.sqrt(2)):
    if len(node.children) == 0:
        print node
        raise ValueError("Empty children")

    L = [n.Q/float(n.N) + C*math.sqrt(2*math.log(sum(n2.N for n2 in node.children))/n.N)
         for n in node.children]
    return np.argmax(L)

from misc.config import main_logger

class UCT(object):
    def __init__(self, N, policy=eps_greedy_policy, seed=None, progressive_widening=False,
                 logger=main_logger):
        self.policy = policy
        self.progressive_widening = progressive_widening
        self.logger = logger #TODO: as decorator
        self.N = N
        self.seed = seed

    def to_graphviz(self, max_depth=10):
        import graphviz as gv
        g = gv.Graph(format='png')
        def construct_graph_dfs(node, depth=0):
            if depth > max_depth:
                return
            g.node(str(node.id), label=self.game.repr(node.state)+"\n"+str(node.Q/float(node.N))+":"+str(node.N))
            for child in node.children:
                g.edge(str(node.id), str(child.id))
                construct_graph_dfs(child,depth+1)
        construct_graph_dfs(self.root)
        return g

    def fit(self, game, state):
        self.game = game
        self.root = UCTNode(id=0, state=state, children=[], Q=0, N=0, \
                            current_action=0, actions=self.game.get_actions(state), parent=None)
        self.max_id = 0
        if self.seed:
            self.rng = np.random.RandomState(self.seed)
        else:
            self.rng = np.random.RandomState()

        self.nodes = {self.game.state_key(self.root.state): self.root}

        for i in range(self.N):
            # Select node
            node = self._tree_policy(self.root)

            # Playout and propagate reward
            if not self.game.is_terminal(node.state):
                delta = self.game.utility(self.game.playout_randomly(node.state, self.rng))
            else:
                delta = self.game.utility(node.state)
            self._propagate(node, delta)

        # Call policy without exploration
        self.best_action = self.root.actions[np.argmax([n.Q/float(n.N) for n in self.root.children])]

        best_path = [self.root]

        node = self.root
        while len(node.children):
            node = node.children[np.argmax([n.Q/float(n.N) for n in node.children])]
            best_path.append(node)

        self.best_path = best_path

    def _expand(self, node):
        if self.progressive_widening:
            return len(node.children) < np.floor(node.N**0.25) + 1 and len(node.actions) > node.current_action
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

               existing_state = self.nodes.get(self.game.state_key(new_state), None)

               if existing_state and len(existing_state.state["ids"]) <= len(node.state["ids"]):
                   print node
                   print existing_state
                   print action
                   raise ValueError("Backward edge")

               if existing_state:
                   node.children.append(existing_state)
                   return existing_state
               else:
                   self.max_id += 1
                   new_node = UCTNode(id=self.max_id, state=new_state, parent=node, N=0, Q=0, \
                                      children=[], current_action=0, actions=self.game.get_actions(new_state))
                   node.children.append(new_node)
                   return new_node
            else:
               node = node.children[self.policy(node)]

            if depth > 100:
                raise ValueError("Cycle in state graph")

        return node

    def _propagate(self, node, delta):
        while node is not None:
            node.N += 1
            # Minus because we are doing argmax always, so if child loses we win
            node.Q += -delta[node.state['player']]
            node = node.parent


def transition_sample(state, action):
    new_state = copy.deepcopy(state)
    new_state["board"][action] = new_state["player"]
    new_state["player"] = (state["player"] + 1)%2
    return new_state

def is_terminal_sample(state):
    return np.all((state["board"] == 0) + (state["board"] == 1)) \
        or sum(utility_sample(state)) != 0

def utility_sample(state):
    def won(A):
        if any(np.array_equal(r , [1,1,1]) for r in A) or \
            any(np.array_equal(c , [1,1,1]) for c in A.T) or \
            (A[0,0] == A[1,1] == A[2,2] == 1) or \
            (A[2,0] == A[1,1] == A[0,2] == 1):
            return 1
        else:
            return 0
    # :P
    rev_board = state["board"].copy()
    rev_board[rev_board==0] = 2
    rev_board[rev_board==1] = 0
    rev_board[rev_board==2] = 1

    if won(rev_board):
        return [1,-1]
    if won(state["board"]):
        return [-1, 1]
    return [0,0]

def get_actions_sample(state):
    return zip(*np.where(((state['board'] == "x"))))

def repr_sample(state):
    return "\n".join(" ".join([str(v) for v in row]) for row in state["board"])

def state_key_sample(state):
    return str(state['board'])

def playout_randomly_sample(state, rng):
    state = copy.deepcopy(state)
    actions = get_actions_sample(state)
    random.shuffle(actions)
    player = state["player"]
    for a in actions:
        state["board"][a] = player
        player = (player + 1)%2
    return state

cross_and_circle = UCTGame(transition=transition_sample, repr=repr_sample, state_key=state_key_sample,
                     playout_randomly=playout_randomly_sample, get_actions=get_actions_sample,\
                     is_terminal=is_terminal_sample, utility=utility_sample
                     )