import tensorflow as tf

from keras.layers import Dense, LeakyReLU, Input, Concatenate, BatchNormalization
from keras.regularizers import l1, l2, l1_l2
from keras.activations import softmax
from keras import Model, Sequential
from keras.utils import plot_model

# TODO: Add weight initialisation
# TODO: Specify training method

"""
Inputs: Patient parameters, current solution, (dose history? avoided by using RNN)
Outputs: Policy - Immunotherapy and Virotherapy dose at given day (Classification)
         Value - Quality of position from +1 to -1 (Regression)
"""


class Learner:

    def __init__(self, param_size=5, state_size=28, policy_size=2, value_size=1):
        self.param_size = param_size
        self.state_size = state_size
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
        x = Dense(self.value_size, use_bias=False, activation='linear', kernel_regularizer=l2())(x)
        return x

    def __policy_head(self, x):
        x = Dense(10, kernel_regularizer=l2())(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dense(self.policy_size, use_bias=False, activation='linear', kernel_regularizer=l2())(x)
        return x

    def __build_model(self):

        # Accept and merge two inputs
        inputs = [Input(shape=(self.param_size,)), Input(shape=(self.state_size,))]
        merged_input = Concatenate()([inputs[0], inputs[1]])

        # Build the core from the merged input
        core = self.__core(merged_input)

        # Build the heads from the core
        value_head = self.__value_head(core)
        policy_head = self.__policy_head(core)

        # Return the final model
        return Model(inputs=inputs, outputs=[value_head, policy_head])

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
