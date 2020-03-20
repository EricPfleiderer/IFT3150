import numpy as np
import random
import tensorflow as tf
from keras.layers import Dense, LeakyReLU, Input, Concatenate, BatchNormalization, Reshape
from keras.regularizers import l2
from keras.optimizers import SGD
from keras import Model
from keras.utils import plot_model
from keras.backend import zeros, shape, equal

from AlphaSilico.src.MCTS import Node, Edge, MCTS
from AlphaSilico.src import config


class Losses:

    @staticmethod
    def softmax_cross_entropy_with_logits(y_true, y_pred):

        pi = y_true
        p = y_pred

        zero = zeros(shape=shape(pi), dtype=tf.float32)
        where = equal(pi, zero)

        negatives = tf.fill(tf.shape(pi), -100.0)
        p = tf.where(where, negatives, p)

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=pi, logits=p)

        return loss


class Learner:

    def __init__(self, learning_rate, input_shapes=((7,), (28,)), policy_size=4, value_size=1):
        """
        Multiple input, twin output model.
        :param learning_rate: Float. Learning rate during training.
        :param input_shapes: Tuple of tuples. Each entry represents the input shape of a head.
        :param policy_size: Int. Policy head output shape.
        :param value_size: Int. Value head output shape.
        """
        self.learning_rate = learning_rate
        self.input_shapes = input_shapes
        self.policy_size = policy_size
        self.value_size = value_size
        self.model = self.__build_model()

    def __core(self, merged_input):
        x = Dense(50, kernel_regularizer=l2())(merged_input)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dense(25, kernel_regularizer=l2())(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        return x

    def __value_head(self, x):
        x = Dense(10, kernel_regularizer=l2())(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dense(self.value_size, use_bias=False, activation='linear', kernel_regularizer=l2(), name='value_head')(x)
        return x

    def __policy_head(self, x):
        x = Dense(10, kernel_regularizer=l2())(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dense(self.policy_size*2, use_bias=False, activation='linear', kernel_regularizer=l2())(x)
        x = Reshape((2, self.policy_size), name='policy_head')(x)
        return x

    def __build_model(self):

        # Accept and merge inputs
        inputs = [Input(shape=input_shape) for input_shape in self.input_shapes]
        merged_input = Concatenate()(inputs)

        # Build the core from the merged input
        core = self.__core(merged_input)

        # Build the heads from the core
        value_head = self.__value_head(core)
        policy_head = self.__policy_head(core)

        # Initialize and compile model
        model = Model(inputs=inputs, outputs=[value_head, policy_head])
        model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': Losses.softmax_cross_entropy_with_logits},
                      optimizer=SGD(lr=self.learning_rate, momentum=config.MOMENTUM),
                      loss_weights={'value_head': 0.5, 'policy_head': 0.5})

        # Return the final model
        return model

    def convert_to_input(self, state):
        pass

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, states, targets, epochs, verbose, validation_split, batch_size):
        return self.model.fit(states, targets, epochs=epochs, verbose=verbose, validation_split=validation_split, batch_size=batch_size)

    def write(self, model, version):
        pass

    def read(self, model, version):
        pass

    def plot_model(self):
        if self.model is not None:
            plot_model(self.model, to_file='outputs/Model_graph.png')


class Agent:

    def __init__(self, name, action_size, mcts_simulations, cpuct, learner):

        """
        Self-reinforcement learning agent. Uses twin headed model and Monte Carlo tree searches according to AlphaZero architecture by deepmind.
        Interfaces with Environment class.
        :param name: String. Agent designation.
        :param action_size: Tuple. Allowed dose intervals for immunotherapy and virotherapy respectively.
        :param mcts_simulations: Int. Number of MCTS simulations.
        :param cpuct: Float. Exploration constant.
        :param learner: Learner class. Interface to neural network.
        """
        self.name = name
        self.action_size = action_size
        self.cpuct = cpuct
        self.mcts_simulations = mcts_simulations
        self.brain = learner

        self.mcts = None
        self.root = None
        self.train_overall_loss = []
        self.train_value_loss = []
        self.train_policy_loss = []
        self.val_overall_loss = []
        self.val_value_loss = []
        self.val_policy_loss = []

    def simulate(self):
        # Move to the leaf node
        leaf, value, done, breadcrumbs = self.mcts.move_to_leaf()

        # Evaluate the leaf node
        value, breadcrumbs = self.evaluate_leaf(leaf, value, done, breadcrumbs)

        # Backfill the value through the tree
        self.mcts.back_fill(leaf, value, breadcrumbs)

    def act(self, state, tau):

        if self.mcts is None or state.id not in self.mcts.tree:
            self.build_MCTS(state)
        else:
            self.change_MCTS_root(state)

        # Run a number of Monte Carlo Tree Searches
        for sim in range(self.mcts_simulations):
            self.simulate()

        #  Get action values
        pi, values = self.get_action_values(1)  # Deterministic play

        # Pick the action
        action, value = self.choose_action(pi, values, tau)

        nextState, _, _ = state.takeAction(action)

        NN_value = -self.get_preds(nextState)[0]

        return action, pi, value, NN_value

    def get_preds(self, state):

        # Predict the leaf
        model_input = np.array([self.brain.convert_to_model_input(state)])

        preds = self.brain.predict(model_input)
        value_array = preds[0]
        logits_array = preds[1]
        value = value_array[0]  # Value head
        logits = logits_array[0]  # Policy head

        # Mask illegal actions
        allowed_actions = state.allowed_actions
        mask = np.ones(logits.shape, dtype=bool)
        mask[allowed_actions] = False
        logits[mask] = -100

        # SOFTMAX
        odds = np.exp(logits)
        probs = odds / np.sum(odds)

        return value, probs, allowed_actions

    def evaluate_leaf(self, leaf, value, done, breadcrumbs):

        if not done:
            value, probs, allowed_actions = self.get_preds(leaf.state)

            probs = probs[allowed_actions]

            for idx, action in enumerate(allowed_actions):
                new_state, _, _ = leaf.state.step(action)
                if new_state.id not in self.mcts.tree:
                    node = Node(new_state)
                    self.mcts.add_node(node)
                else:
                    node = self.mcts.tree[new_state.id]
                newEdge = Edge(leaf, node, probs[idx], action)
                leaf.edges.append((action, newEdge))

        return value, breadcrumbs

    def get_action_values(self, tau):
        edges = self.mcts.root.edges
        pi = np.zeros(self.action_size, dtype=np.integer)
        values = np.zeros(self.action_size, dtype=np.float32)

        for action, edge in edges:
            pi[action] = pow(edge.stats['N'], 1 / tau)
            values[action] = edge.stats['Q']

        pi = pi / (np.sum(pi) * 1.0)
        return pi, values

    @staticmethod
    def choose_action(pi, values, tau):

        # Deterministic play
        if tau == 0:
            actions = np.argwhere(pi == max(pi))
            action = random.choice(actions)[0]

        # Random play
        else:
            action_idx = np.random.multinomial(1, pi)
            action = np.where(action_idx == 1)[0][0]

        value = values[action]

        return action, value

    def replay(self, ltmemory):

        for i in range(config.TRAINING_LOOPS):
            minibatch = random.sample(ltmemory, min(config.BATCH_SIZE, len(ltmemory)))  # Sample mini batch from long term memory

            training_states = np.array([self.brain.convert_to_model_input(row['state']) for row in minibatch])
            training_targets = {'value_head': np.array([row['value'] for row in minibatch]), 'policy_head': np.array([row['AV'] for row in minibatch])}

            fit = self.brain.fit(training_states, training_targets, epochs=config.EPOCHS, verbose=1, validation_split=0, batch_size=32)

            self.train_overall_loss.append(round(fit.history['loss'][config.EPOCHS - 1], 4))
            self.train_value_loss.append(round(fit.history['value_head_loss'][config.EPOCHS - 1], 4))
            self.train_policy_loss.append(round(fit.history['policy_head_loss'][config.EPOCHS - 1], 4))

    def predict(self, input_to_model):
        return self.brain.predict(input_to_model)

    def build_MCTS(self, state):
        self.root = Node(state)
        self.mcts = MCTS(self.root, self.cpuct)

    def change_MCTS_root(self, state):
        self.mcts.root = self.mcts.tree[state.get_id()]
