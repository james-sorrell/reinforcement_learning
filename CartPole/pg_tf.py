import gym
import os
import sys

import config

import numpy as np
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from cartpole.q_learning_bins import plot_running_avg


# so you can test different architectures
class HiddenLayer:
  def __init__(self, M1, M2, f=tf.nn.tanh, use_bias=True):
    self.W = tf.Variable(tf.random_normal(shape=(M1, M2)))
    self.use_bias = use_bias
    if use_bias:
      self.b = tf.Variable(np.zeros(M2).astype(np.float32))
    self.f = f

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
    M1 = D
    for M2 in hidden_layer_sizes:
      layer = HiddenLayer(M1, M2)
      self.layers.append(layer)
      M1 = M2

    # final layer
    layer = HiddenLayer(M1, K, tf.nn.softmax, use_bias=False)
    self.layers.append(layer)

    # inputs and targets
    self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')
    self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')

    # calculate output and cost
    Z = self.X
    for layer in self.layers:
      Z = layer.forward(Z)
    # Proability distribution over our actions for
    # given state, as predicted by NN defined as a 
    # function that we can assign and execute externally
    # make them 1-D
    
    p_a_given_s = Z
    # action_scores = Z
    # p_a_given_s = tf.nn.softmax(action_scores)
    # self.action_scores = action_scores
    self.predict_op = p_a_given_s

    # self.one_hot_actions = tf.one_hot(self.actions, K)

    # This is a way to index p_a_given_s
    # using a one-hot matrix for the actions
    # that we took. reduce sum is a convenient
    # way for us to remove the zero'd out dimensions
    # we then log this because the cost function
    # for policy gradient method is logged.
    selected_probs = tf.log(
      tf.reduce_sum(
        p_a_given_s * tf.one_hot(self.actions, K),
        reduction_indices=[1]
      )
    )

    # We negate this because we want to maximise this value
    # however, tensorflows optimisers only minimise.
    cost = -tf.reduce_sum(self.advantages * selected_probs)
    # Here we have multiple optimisers that we could choose with different
    # learning parameters that we can choose.
    self.train_op = tf.train.AdamOptimizer(1e-1).minimize(cost)
    # self.train_op = tf.train.AdagradOptimizer(1e-1).minimize(cost)
    # self.train_op = tf.train.MomentumOptimizer(1e-4, momentum=0.9).minimize(cost)
    # self.train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)

  def set_session(self, session):
    self.session = session

  def partial_fit(self, X, actions, advantages):
    # States, actions & advantages set to correct
    # array sizes
    X = np.atleast_2d(X)
    actions = np.atleast_1d(actions)
    advantages = np.atleast_1d(advantages)
    # Then we call the train operation
    self.session.run(
      self.train_op,
      feed_dict={
        self.X: X,
        self.actions: actions,
        self.advantages: advantages,
      }
    )

  def predict(self, X):
    X = np.atleast_2d(X)
    return self.session.run(self.predict_op, feed_dict={self.X: X})

  # Note that we don't use epsilon greedy here! This is because policy
  # gradient is a probabalistic methodology which converges on an 
  # optimal solution without the need for epsilon greedy.
  def sample_action(self, X):
    p = self.predict(X)[0]
    return np.random.choice(len(p), p=p)


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

    # inputs and targets
    self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')

    # calculate output and cost
    Z = self.X
    for layer in self.layers:
      Z = layer.forward(Z)
    Y_hat = tf.reshape(Z, [-1]) # the output
    # Predict operation
    self.predict_op = Y_hat

    cost = tf.reduce_sum(tf.square(self.Y - Y_hat))

    # Training operator with different potential optimisers
    # self.train_op = tf.train.AdamOptimizer(1e-2).minimize(cost)
    # self.train_op = tf.train.MomentumOptimizer(1e-2, momentum=0.9).minimize(cost)
    self.train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)

  def set_session(self, session):
    self.session = session

  def partial_fit(self, X, Y):
    X = np.atleast_2d(X)
    Y = np.atleast_1d(Y)
    self.session.run(self.train_op, feed_dict={self.X: X, self.Y: Y})

  def predict(self, X):
    X = np.atleast_2d(X)
    return self.session.run(self.predict_op, feed_dict={self.X: X})


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
  while not done and iters < 2000:
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
  # tf init
  init = tf.global_variables_initializer()
  session = tf.InteractiveSession()
  session.run(init)
  # pvmodels init
  pmodel.set_session(session)
  vmodel.set_session(session)
  gamma = 0.99

  if 'monitor' in sys.argv:
    record=True

  save_folder = config.getSaveFolder('cartpole', os.path.basename(__file__).split('.')[0])
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

