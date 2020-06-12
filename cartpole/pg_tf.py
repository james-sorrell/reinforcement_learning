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
from datetime import datetime
from cartpole.q_learning_bins import plot_running_avg


# so you can test different architectures
class HiddenLayer:
  def __init__(self, M1, M2, f=keras.activations.tanh, use_bias=True):
    #print("\n\tM1: {}, M2: {}\n".format(M1, M2))
    self.W = tf.Variable(tf.random.normal(shape=(M1, M2)))
    self.use_bias = use_bias
    if use_bias:
      self.b = tf.Variable(np.zeros(M2).astype(np.float32))
    self.f = f

  @tf.function
  def forward(self, X):
    # forward mat mul
    a = tf.matmul(X, self.W)
    # bias
    if self.use_bias:
      a += self.b
    # activation function
    return self.f(a)

# approximates pi(a | s)
class PolicyModel:
  def __init__(self, D, K, hidden_layer_sizes):
    # Create the graph -> K = num actions
    self.layers = []
    self.K = K
    M1 = D
    for M2 in hidden_layer_sizes:
      layer = HiddenLayer(M1, M2)
      self.layers.append(layer)
      M1 = M2

    # final layer
    layer = HiddenLayer(M1, K, keras.activations.softmax, use_bias=False)
    self.layers.append(layer)

    # add layers to keras model
    self.model = keras.Sequential()
    for layer in self.layers:
      self.model.add(keras.layers.Lambda(layer.forward))
    # define optimizer
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-1)

  def partial_fit(self, X, actions, advantages):
    X = np.atleast_2d(X)
    actions = np.atleast_1d(actions)
    advantages = np.atleast_1d(advantages)
    with tf.GradientTape() as t:
      # This is a way to index p_a_given_s
      # using a one-hot matrix for the actions
      # that we took. reduce sum is a convenient
      # way for us to remove the zero'd out dimensions
      # we then log this because the cost function
      # for policy gradient method is logged.
      selected_probs = tf.math.log(
        tf.reduce_sum(self.predict(X) * keras.backend.one_hot(actions, self.K), axis=1)
      )
      # We negate this because we want to maximise this value
      # however, tensorflows optimisers only minimise.
      cost = -tf.reduce_sum(advantages * selected_probs)
    grads = t.gradient(cost, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

  def predict(self, X):
    # Proability distribution over our actions for
    # given state, as predicted by NN defined as a 
    # function that we can assign and execute externally
    # make them 1-D
    return self.model(np.atleast_2d(X)) # self.session.run(self.predict_op, feed_dict={self.X: X})

  # Note that we don't use epsilon greedy here! This is because policy
  # gradient is a probabalistic methodology which converges on an 
  # optimal solution without the need for epsilon greedy.
  def sample_action(self, X):
    p = self.predict(X)[0]
    # Samples from p accoirding to provided proability distribution
    dist = tfp.distributions.Categorical(p)
    #print(dist.sample())
    return dist.sample().numpy()

# Approximates V(s)
class ValueModel:
  def __init__(self, D, hidden_layer_sizes):
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

    # create trainable keras model
    self.model = keras.Sequential()
    for layer in self.layers:
      self.model.add(keras.layers.Lambda(layer.forward))

    self.optimizer = tf.keras.optimizers.SGD(1e-4)

  def partial_fit(self, X, Y):
    X = np.atleast_2d(X)
    Y = np.atleast_1d(Y)
    with tf.GradientTape() as t:
      cost = tf.reduce_sum(tf.square(Y - self.model(X)))
    grads = t.gradient(cost, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

  def predict(self, X):
    return self.model(np.atleast_2d(X))

def play_one_td(env, pmodel, vmodel, gamma):
  observation = env.reset()
  done = False
  totalreward = 0
  iters = 0

  while not done:
    # if we reach 2000, just quit, don't want this going forever
    # the 200 limit seems a bit early
    action = pmodel.sample_action(observation)
    prev_observation = observation
    observation, reward, done, _ = env.step(action)

    # if done:
    #   reward = -200

    # update the models
    V_next = vmodel.predict(observation)[0]
    G = reward + gamma*V_next
    advantage = G - vmodel.predict(prev_observation)
    pmodel.partial_fit(prev_observation, action, advantage)
    vmodel.partial_fit(prev_observation, G)

    if reward == 1: # if we changed the reward to -200
      totalreward += reward
    iters += 1

  return totalreward



def play_one_mc(env, pmodel, vmodel, gamma):
  observation = env.reset()
  done = False
  totalreward = 0
  iters = 0

  states = []
  actions = []
  rewards = []

  reward = 0
  while not done and iters < 10000:
    # if we reach 2000, just quit, don't want this going forever
    # the 200 limit seems a bit early
    action = pmodel.sample_action(observation)

    states.append(observation)
    actions.append(action)
    rewards.append(reward)

    observation, reward, done, _ = env.step(action)

    # Punish for finishing early
    if done:
      reward = -200

    if reward == 1: # if we changed the reward to -200
      totalreward += reward
    iters += 1

  # save the final (s,a,r) tuple
  action = pmodel.sample_action(observation)
  states.append(observation)
  actions.append(action)
  rewards.append(reward)

  returns = []
  advantages = []
  G = 0
  for s, r in zip(reversed(states), reversed(rewards)):
    returns.append(G)
    advantages.append(G - vmodel.predict(s)[0])
    G = r + gamma*G
  returns.reverse()
  advantages.reverse()

  # update the models
  pmodel.partial_fit(states, actions, advantages)
  vmodel.partial_fit(states, returns)

  return totalreward


def main():
  env = gym.make('CartPoleLong-v0')
  D = env.observation_space.shape[0]
  K = env.action_space.n
  pmodel = PolicyModel(D, K, [])
  vmodel = ValueModel(D, [10])
  gamma = 0.99

  record=False
  if 'monitor' in sys.argv:
    record=True

  save_folder = config.getSaveFolder(os.path.basename(os.path.dirname(__file__)), os.path.basename(__file__).split('.')[0])
  if record is True:
    env = wrappers.Monitor(env, os.path.join(save_folder, 'monitor'))

  N = 1000
  totalrewards = np.empty(N)
  for n in range(N):
    totalreward = play_one_mc(env, pmodel, vmodel, gamma)
    totalrewards[n] = totalreward
    if n % 100 == 0:
      print("Episode: {}, Total Reward: {}, Average Reward (100): {}".format(n, totalreward, totalrewards[max(0, n-100):(n+1)].mean()))
  print("Average reward for last 100 episodes:", totalrewards[-100:].mean())
  print("Total steps:", -totalrewards.sum())

  config.plot_total_reward(totalrewards, save_folder, record)
  config.plot_running_avg(totalrewards, save_folder, record)
  plot_running_avg(totalrewards)


if __name__ == '__main__':
  main()

