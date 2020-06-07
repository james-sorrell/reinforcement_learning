# https://deeplearningcourses.com/c/cutting-edge-artificial-intelligence
import numpy as np
import tensorflow as tf
from tensorflow import keras


def sample(logits):
    """ Sample from logits distribution helper """
    noise = tf.random.uniform(tf.shape(logits))
    return tf.argmax(logits - tf.math.log(-tf.math.log(noise)), 1)


def conv(inputs, nf, ks, strides, gain=np.sqrt(2)):
    """ Conv Layer Helper """
    return keras.layers.Conv2D(filters=nf, kernel_size=ks,
                            strides=strides, activation=tf.nn.relu,
                            kernel_initializer=keras.initializers.Orthogonal(gain=gain))(inputs)


def dense(inputs, units, activation=tf.nn.relu, gain=1.0):
    """ Dense Layer Helper """
    return keras.layers.Dense(units=units, activation=activation,
                           kernel_initializer=keras.initializers.Orthogonal(gain=gain))(inputs)


def build_feature_extractor(input_):
  # scale the inputs from 0..255 to 0..1
  input_ = tf.cast(input_, dtype=tf.float32) / 255.0
  # conv layers
  x = conv(input_, 32, 8, 4)
  x = conv(x, 64, 4, 2)
  x = conv(x, 64, 3, 1)
  x = keras.layers.Flatten()(x)
  return dense(x, 512, gain=np.sqrt(2))


class CNN:

    def __init__(self, ob_space, ac_space, nenv, nsteps, nstack, lr, alpha, epsilon, ent_coef, vf_coef, max_grad_norm):

        # nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        #print(nbatch, nh, nw, nc*nstack)
        #ob_shape = (nbatch, nh, nw, nc * nstack)

        self.X = keras.Input(shape=[nh, nw, nc*nstack], dtype=tf.uint8, name="X")  # obs
        features = build_feature_extractor(self.X)
        # Policy
        pi = dense(features, ac_space.n, activation=None)
        # Value
        vf = dense(features, 1, activation=None)
        self.network = tf.keras.Model(inputs=self.X, outputs=[pi, vf])
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr, rho=alpha, epsilon=epsilon)
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

    def step(self, ob):
        pi, vf =  self.network(ob)
        return sample(pi), vf[:,0]

    def value(self, ob):
        _, vf =  self.network(ob)
        return vf[:,0]

    def cat_entropy(self, logits):
        a0 = logits - tf.reduce_max(logits, 1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, 1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), 1)

    def train(self, states, actions, advantages, rewards):
        # Compute Gradients
        with tf.GradientTape() as t:
            pi, vf = self.network(states)
            neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pi, labels=actions)
            pg_loss = tf.reduce_mean(advantages * neglogpac)
            vf_loss = tf.reduce_mean(tf.math.squared_difference(tf.squeeze(vf), rewards) / 2.0)
            entropy = tf.reduce_mean(self.cat_entropy(pi))
            loss = pg_loss - entropy * self.ent_coef + vf_loss * self.vf_coef
        grads = t.gradient(loss, self.network.trainable_weights)
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_weights))
        return pg_loss, vf_loss, entropy
