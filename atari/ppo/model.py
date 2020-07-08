import tensorflow as tf
from tensorflow import keras

def network(input):
    out = keras.layers.Conv2D(filters=32, kernel_size=8, 
                        stride=4, activation_fn=tf.nn.relu)(input)
    out = keras.layers.Conv2D(filters=64, kernel_size=4,
                        stride=2, activation_fn=tf.nn.relu)(out)
    out = keras.layers.Conv2D(filters=64, kernel_size=3,
                        stride=1, activation_fn=tf.nn.relu)(out)
    out = keras.layers.flatten(out)
    return keras.layers.Dense(units=512, activation_fn=tf.nn.relu)(out)

def value_tail(shared):
    return tf.squeeze(layers.fully_connected(shared, num_outputs=1, activation_fn=None), axis=1)

def policy_tail(shared):
    return tf.nn.softmax(layers.fully_connected(shared, num_outputs=num_actions, activation_fn=None), axis=1)
