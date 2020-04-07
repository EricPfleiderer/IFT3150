

# Third-party
import numpy as np

# Local
from AlphaSilico.src import config


class Node:

    def __init__(self, state):

        """
        Node object in Monte Carlo Tree.
        :param state: state instance.
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
        :param node_in: State instance. Parent.
        :param node_out: State instance. Children.
        :param prior: Float. Prior probabilities over actions.
        :param action: List. Action space.
        """

        self.id = node_in.state.id + '|' + node_out.state.id
        self.node_in = node_in
        self.node_out = node_out
        self.action = action

        self.stats = {
            'N': 0,  # Visit count
            'W': 0,  # Total action-value
            'Q': 0,  # Mean action-value
            'P': prior,  # Prior probability of choosing the move
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
    def backup(value, breadcrumbs):
        for edge in breadcrumbs:
            edge.stats['N'] = edge.stats['N'] + 1
            edge.stats['W'] = edge.stats['W'] + value
            edge.stats['Q'] = edge.stats['W'] / edge.stats['N']

    def select(self):

        """
        Used to traverse a built MCTS tree a single time from root to leaf.
        :return: A leaf node. Returns root if it has no children.
        """

        breadcrumbs = []  # Ordered list of visited edges
        current_node = self.root

        done = 0
        value = 0

        simulation_action = None
        simulation_edge = None

        while not current_node.is_leaf() and not done:

            max_QU = -float('inf')  #

            # Adding Dirichlet noise to the root priors of the root node
            if current_node == self.root:
                epsilon = config.EPSILON
                nu = np.random.dirichlet([config.ALPHA] * len(current_node.edges))
            else:
                epsilon = 0
                nu = [0] * len(current_node.edges)

            Nb = 0
            for action, edge in current_node.edges:
                Nb = Nb + edge.stats['N']

            # Choose action
            for idx, (action, edge) in enumerate(current_node.edges):

                U = self.cpuct * ((1 - epsilon) * edge.stats['P'] + epsilon * nu[idx]) * np.sqrt(Nb) / (1 + edge.stats['N'])  # Exploration term

                Q = edge.stats['Q']  # Exploitation term

                if Q + U > max_QU:  # Systematic bias for first encounter (>) or last encounter (>=) if multiple max entries (initializations to 0)
                    max_QU = Q + U
                    simulation_action = action
                    simulation_edge = edge

            current_node = simulation_edge.node_out  # Travel to the next node
            breadcrumbs.append(simulation_edge)  # Keep track of visited edges

            # Get statistics of the next leaf
            new_state, value, done = current_node.state.take_action(simulation_action)  # (SLOW)

        return current_node, value, done, breadcrumbs


