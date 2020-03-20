

# Third-party
import numpy as np

# Local
from AlphaSilico.src import config


class Node:

    def __init__(self, state):

        """
        Node object in Monte Carlo Tree.
        :param state: State instance.
        """

        self.state = state
        self.id = state.get_id()
        self.edges = []

    def is_leaf(self):
        return len(self.edges) == 0


class Edge:

    def __init__(self, node_in, node_out, prior, action):

        """
        Edge object in Monte Carlo Tree.
        :param node_in:
        :param node_out:
        :param prior:
        :param action:
        """

        self.id = node_in.state.id + '|' + node_out.state.id
        self.node_in = node_in
        self.node_out = node_out
        self.action = action

        self.stats = {
            'N': 0,
            'W': 0,
            'Q': 0,
            'P': prior,
        }


class MCTS:

    def __init__(self, root, cpuct):
        """
        :param root: Node instance. Root node of the MCTS tree.
        :param cpuct: Float. Constant that determines the level of exploration during look-ahead.
        """

        self.cpuct = cpuct
        self.root = root
        self.tree = {}
        self.add_node(root)

    def __len__(self):
        return len(self.tree)

    def add_node(self, node):
        self.tree[node.id] = node

    @staticmethod
    def back_up(value, breadcrumbs):

        for edge in breadcrumbs:
            edge.stats['N'] = edge.stats['N'] + 1
            edge.stats['W'] = edge.stats['W'] + value
            edge.stats['Q'] = edge.stats['W'] / edge.stats['N']

    def move_to_leaf(self):

        breadcrumbs = []
        current_node = self.root

        done = 0
        value = 0

        while not current_node.is_leaf():

            maxQU = -float('inf')  # 99999

            if current_node == self.root:
                epsilon = config.EPSILON
                nu = np.random.dirichlet([config.ALPHA] * len(current_node.edges))
            else:
                epsilon = 0
                nu = [0] * len(current_node.edges)

            Nb = 0
            for action, edge in current_node.edges:
                Nb = Nb + edge.stats['N']

            simulation_action = None
            simulation_edge = None

            for idx, (action, edge) in enumerate(current_node.edges):

                U = self.cpuct * ((1 - epsilon) * edge.stats['P'] + epsilon * nu[idx]) * np.sqrt(Nb) / (1 + edge.stats['N'])

                Q = edge.stats['Q']

                if Q + U > maxQU:
                    maxQU = Q + U
                    simulation_action = action
                    simulation_edge = edge

            new_state, value, done = current_node.state.step(simulation_action)
            current_node = simulation_edge.node_out
            breadcrumbs.append(simulation_edge)

        return current_node, value, done, breadcrumbs


