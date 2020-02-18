import tensorflow as tf
from tensorflow import concat
from tensorflow.keras.layers import Layer, Dense, LeakyReLU
from tensorflow.keras import Model
from tensorflow.keras.activations import softmax

# TODO: Add weight initialisation
# TODO: Specify training method


class Vision(Layer):
    def __init__(self, weights1=20, weights2=10):
        super(Vision, self).__init__()
        self.layer1 = Dense(weights1)
        self.layer2 = Dense(weights2)

    def __call__(self, inputs):
        #self.add_loss(loss_func)
        x = self.layer1(inputs) 
        x = LeakyReLU(x)
        x = self.layer2(x) 
        x = LeakyReLU(x)
        return x


class ValueHead(Layer):
    def __init__(self, weights=10):
        super(ValueHead, self).__init__()
        self.layer1 = Dense(weights)
        self.out = Dense(1)

    def __call__(self, inputs):
        x = self.layer1(inputs)
        x = softmax(x)
        x = self.out(x)
        return x


class PolicyHead(Layer):
    def __init__(self, weights1=20, weights2=10):
        super(PolicyHead, self).__init__()
        self.layer1 = Dense(weights1)
        self.layer2 = Dense(weights2)
        self.out = Dense(2)

    def __call__(self, inputs):
        x = self.layer1(inputs)
        x = LeakyReLU(x)
        x = self.layer2(x)
        x = softmax(x)
        x = self.out(x)
        return x


class Learner(Model):

    def __init__(self):
        super(Model, self).__init__()
        self.vision = Vision()
        self.value_head = ValueHead()
        self.policy_head = PolicyHead()

    def __call__(self, y):
        y = self.vision(y)
        yv = self.value_head(y)
        yp = self.policy_head(y)
        output = concat([yv, yp], axis=1)
        return output 

