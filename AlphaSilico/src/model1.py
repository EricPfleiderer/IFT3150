import tensorflow as tf
# TODO: Add weight initialisation
# TODO: Specify training method

class Vision(tf.keras.layers.Layer):
    def __init__(self, weights1=20, weights2=10):
        super(Vision, self).__init__()
        self.layer1 = tf.keras.layers.Dense(weights1)
        self.layer2 = tf.keras.layers.Dense(weights2)
        

    def call(self, inputs):
        #self.add_loss(loss_func)
        x = self.layer1(inputs) 
        x = tf.nn.leaky_relu(x)
        x = self.layer2(x) 
        x = tf.nn.softmax(x)
        return x


class ValueHead(tf.keras.layers.Layer):
    def __init__(self, weights=10):
        super(ValueHead, self).__init__()
        self.layer1 = tf.keras.layers.Dense(weights)
        self.out = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.layer1(inputs)
        x = tf.nn.softmax(x)
        x = self.out(x)
        return x


class PolicyHead(tf.keras.layers.Layer):
    def __init__(self, weights1=20, weights2=10):
        super(PolicyHead, self).__init__()
        self.layer1 = tf.keras.layers.Dense(weights1)
        self.layer2 = tf.keras.layers.Dense(weights2)
        self.out = tf.keras.layers.Dense(2)

    def call(self, inputs):
        x = self.layer1(inputs)
        x = tf.nn.leaky_relu(x)
        x = self.layer2(x)
        x = tf.nn.softmax(x)
        x = self.out(x)
        return x


class Model(tf.keras.Model):

    def __init__(self):
        super(Model, self).__init__()
        self.vision = Vision()
        self.value_head = ValueHead()
        self.policy_head = PolicyHead()


    def call(self, y):
        y = self.vision(y)
        yv = self.value_head(y)
        yp = self.policy_head(y)
        output = tf.concat([yv, yp], axis=1)
        return output 

