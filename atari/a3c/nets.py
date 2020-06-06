import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

def build_feature_extractor(input_):
  # scale the inputs from 0..255 to 0..1
  input_ = tf.cast(input_, dtype=tf.float32) / 255.0
  # conv layers
  conv1 = keras.layers.Conv2D(
    filters=16,
    kernel_size=8,
    strides=4,
    activation='relu')(input_)
  conv2 = keras.layers.Conv2D(
    filters=32,
    kernel_size=4,
    strides=2,
    activation='relu')(conv1)
  # image -> feature vector
  flat = keras.layers.Flatten()(conv2)
  # dense layer
  return keras.layers.Dense(units=256)(flat)

class PolicyNetwork:
  def __init__(self, num_outputs, features, input_tensor, reg=0.01):
    self.reg = reg
    self.num_outputs = num_outputs
    # Graph inputs
    # Use shared feature layer
    self.logits = keras.layers.Dense(num_outputs, activation=None)(features)
    # training
    self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.000025, rho=0.99, epsilon=1e-6)
    self.network = tf.keras.Model(inputs=input_tensor, outputs=self.logits)

  def sample_action(self, state):
    logits = self.network(tf.expand_dims(state,0))
    # we'll need these later for running gradient descent steps
    cdist = tfp.distributions.Categorical(logits)
    return cdist.sample()

  def compute_gradients(self, states, advantage, actions):
    """ Calculated gradients for training """
    with tf.GradientTape() as t:
      logits = self.network(states)
      self.probs = keras.backend.softmax(logits)
      # Add regularization to increase exploration
      self.entropy = -tf.reduce_sum(self.probs * tf.math.log(self.probs), axis=1)
      # Get the predictions for the chosen actions only
      batch_size = tf.shape(states)[0]
      gather_indices = tf.range(batch_size) * tf.shape(self.probs)[1] + actions
      self.selected_action_probs = tf.gather(tf.reshape(self.probs, [-1]), gather_indices)
      self.loss = tf.math.log(self.selected_action_probs) * advantage + self.reg * self.entropy
      self.loss = -tf.reduce_sum(self.loss, name="loss")
    grads = t.gradient(self.loss, self.network.trainable_weights)
    return grads

  def train(self, grads):
    """ Allows you to pass in gradients computed elsewhere and train them with local
    network variables """
    grads_and_vars = zip(grads, self.network.trainable_weights)
    self.optimizer.apply_gradients(grads_and_vars)

class ValueNetwork:
  def __init__(self, features, input_tensor):
    # Placeholders for our input
    # After resizing we have 4 consecutive frames of size 84 x 84
    # The TD target value
    #self.targets = keras.Input(shape=[None], dtype=tf.float32, name="y")
    # Shared feature layers are passed in through ValueNetwork constructor
    self.output = keras.layers.Dense(units=1, activation=None)(features)
    # training
    self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.000025, rho=0.99, epsilon=1e-6)
    self.network = tf.keras.Model(inputs=input_tensor, outputs=self.output)

  def sample_action(self, state):
    """ Sample one action from one state """
    output = self.network(tf.expand_dims(state,0))
    # we'll need these later for running gradient descent steps
    return tf.squeeze(output, axis=1)

  def compute_gradients(self, states, targets):
    with tf.GradientTape() as t:
      output = self.network(states)
      self.vhat = tf.squeeze(output, axis=1)
      self.loss = tf.math.squared_difference(self.vhat, targets)
      self.loss = tf.reduce_sum(self.loss)
    grads = t.gradient(self.loss, self.network.trainable_weights)
    return grads

  def train(self, grads):
    """ Allows you to pass in gradients computed elsewhere and train them with local
    network variables """
    grads_and_vars = zip(grads, self.network.trainable_weights)
    self.optimizer.apply_gradients(grads_and_vars)

# Should use this to create networks
# to ensure they're created in the correct order
def create_networks(num_outputs):
  states = keras.Input(shape=[84, 84, 4], dtype=tf.uint8, name="X")
  features = build_feature_extractor(states)
  policy_network = PolicyNetwork(num_outputs=num_outputs, features=features, input_tensor=states)
  value_network = ValueNetwork(features=features, input_tensor=states)
  return policy_network, value_network
