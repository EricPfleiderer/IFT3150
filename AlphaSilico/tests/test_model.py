from AlphaSilico.src.model1 import Model
import tensorflow as tf

def test_model():
    y = tf.ones([1, 10])
    model = Model()
    output = model(y)
    print(output)

    assert 0 == 1


