import gym
import os
import sys
import config
import numpy as np 
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
import matplotlib.pyplot as plt
from gym import wrappers
from mountaincart.q_learning_rbf import FeatureTransformer, plot_cost_to_go

# https://github.com/openai/gym/wiki/MountainCarContinuous-v0
# Reward is 100 for reaching the target of the hill on the right hand side,
# minus the squared sum of actions from start to goal.
# This reward function raises an exploration challenge, 
# because if the agent does not reach the target soon enough, 
# it will figure out that it is better not to move, 
# and won't find the target anymore.
# Note that this reward is unusual with respect to most published work,
# where the goal was to reach the target as fast as possible,
# hence favouring a bang-bang strategy.

class HiddenLayer:
  def __init__(self, M1, M2, f=tf.nn.tanh, use_bias=True, zeros=False):
    if zeros:
      W = np.zeros((M1, M2)).astype(np.float32)
      self.W = tf.Variable(W)
    else:
        # Xavier Initialisation
        W = tf.random.normal(shape=(M1, M2)) * np.sqrt(2. / M1, dtype=np.float32)
    self.W = tf.Variable(W)

    self.use_bias = use_bias
    if use_bias:
      self.b = tf.Variable(np.zeros(M2).astype(np.float32))

    self.f = f

  @tf.function
  def forward(self, X):
    a = tf.matmul(X, self.W)
    if self.use_bias:
      a += self.b
    return self.f(a)

# approximates pi(a | s)
class PolicyModel:
  def __init__(self, D, ft, hidden_layer_sizes=[]):
    self.ft = ft

    ##### hidden layers #####
    M1 = D
    self.hidden_layers = []
    for M2 in hidden_layer_sizes:
      layer = HiddenLayer(M1, M2)
      self.hidden_layers.append(layer)
      M1 = M2

    # final layer, 1 ensures we recieve scalar
    # regression based mean estimation from input vector

    self.mean_layer = HiddenLayer(M1, 1, lambda x: x, use_bias=False, zeros=True)
    # final layer -> Variance needs to be greater than zero
    # hence we use softplus as the activation, 1 ensures we recieve scalar
    # regression based variance estimation
    self.stdv_layer = HiddenLayer(M1, 1, tf.nn.softplus, use_bias=False, zeros=False)

    self.X = keras.Input(shape=(D), dtype=tf.float32,)
    # create trainable keras model
    #self.model = keras.Sequential()
    z = self.X
    for layer in self.hidden_layers:
     z = keras.layers.Lambda(layer.forward)(z)
    # mean, std, make them 1-D
    mean = keras.layers.Lambda(self.mean_layer.forward)(z)
    std = keras.layers.Lambda(self.stdv_layer.forward)(z)
    std = keras.layers.Lambda(lambda x: x + 1e-5)(std) # smoothing
    mean = keras.layers.Flatten()(mean)
    std = keras.layers.Flatten()(std) 
    self.model = keras.Model(inputs=[self.X], outputs=[mean, std])

    self.optimizer = tf.keras.optimizers.Adam(1e-4)

  def set_session(self, session):
    self.session = session

  def partial_fit(self, X, actions, advantages):

    actions = np.atleast_1d(actions)
    advantages = np.atleast_1d(advantages)
    with tf.GradientTape() as t:
      norm = self.unclipped_predict(X)
      log_probs = norm.log_prob(actions)
      cost = -tf.reduce_sum(advantages * log_probs + 0.1*norm.entropy())

    grads = t.gradient(cost, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

  def unclipped_predict(self, X):
    """ This function exists because the train uses
    the unclipped norm """
    X = np.atleast_2d(X)
    X = self.ft.transform(X)
    mean, std = self.model(X)
    return tfp.distributions.Normal(mean, std)
    

  def predict(self, X):
    norm = self.unclipped_predict(X)
    return tf.clip_by_value(norm.sample(), -1, 1)

  def sample_action(self, X):
    p = self.predict(X)[0]
    return p


# approximates V(s)
class ValueModel:
  def __init__(self, D, ft, hidden_layer_sizes=[]):
    self.ft = ft
    self.costs = []

    # create the graph
    self.layers = []
    M1 = D
    for M2 in hidden_layer_sizes:
      layer = HiddenLayer(M1, M2)
      self.layers.append(layer)
      M1 = M2

    # final layer
    layer = HiddenLayer(M1, 1, lambda x: x)
    self.layers.append(layer)

    # inputs and targets
    #self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    #self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')

    # add layers to keras model
    self.model = keras.Sequential()
    for layer in self.layers:
      self.model.add(keras.layers.Lambda(layer.forward))
    self.model.add(keras.layers.Flatten())
    # define optimizer
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-1)

    #cost = tf.reduce_sum(tf.square(self.Y - Y_hat))
    #self.cost = cost
    #self.train_op = tf.train.AdamOptimizer(1e-1).minimize(cost)

  def set_session(self, session):
    self.session = session

  def partial_fit(self, X, Y):
    Y = np.atleast_1d(Y)
    with tf.GradientTape() as t:
      cost = tf.reduce_sum(tf.square(Y - self.predict(X)))
    grads = t.gradient(cost, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    self.costs.append(cost)

  def predict(self, X):
    X = np.atleast_2d(X)
    X = self.ft.transform(X)
    return self.model(X)


def play_one_td(env, pmodel, vmodel, gamma):
  observation = env.reset()
  done = False
  totalreward = 0
  iters = 0

  while not done and iters < 10000:
    # if we reach 10000, just quit, don't want this going forever
    # the 200 limit seems a bit early
    action = pmodel.sample_action(observation)
    prev_observation = observation
    observation, reward, done, _ = env.step([action])

    totalreward += reward

    # update the models
    # reshape observation because openai changes its shape
    observation = observation.reshape([-1])    
    V_next = vmodel.predict(observation)
    G = reward + gamma*V_next
    advantage = G - vmodel.predict(prev_observation)
    pmodel.partial_fit(prev_observation, action, advantage)
    vmodel.partial_fit(prev_observation, G)

    iters += 1

  return totalreward, iters

def main():
  env = gym.make('MountainCarContinuous-v0')
  ft = FeatureTransformer(env, n_components=100)
  D = ft.dimensions
  pmodel = PolicyModel(D, ft, [])
  vmodel = ValueModel(D, ft, [])
  gamma = 0.95

  record = False
  if 'monitor' in sys.argv:
    record=True

  save_folder = config.getSaveFolder(os.path.basename(os.path.dirname(__file__)), os.path.basename(__file__).split('.')[0])
  if record is True:
    env = wrappers.Monitor(env, os.path.join(save_folder, 'monitor'))

  N = 50
  totalrewards = np.empty(N)
  for n in range(N):
    totalreward, num_steps = play_one_td(env, pmodel, vmodel, gamma)
    totalrewards[n] = totalreward
    if n % 1 == 0:
      print("Episode: {}, Total Reward: {:.2f}, Num Steps: {}".format(n, totalreward, num_steps))
  print("Average reward over 100 episodes with best models: {}".format(totalrewards[-100:].mean()))

  config.plot_total_reward(np.array(totalrewards), save_folder, record)
  config.plot_running_avg(np.array(totalrewards), save_folder, record)
  plot_cost_to_go(env, vmodel, save_folder, record)


if __name__ == '__main__':
  main()
