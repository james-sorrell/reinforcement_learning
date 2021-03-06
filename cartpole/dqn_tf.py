import gym
import os
import sys

import config

import numpy as np
import tensorflow as tf 
from tensorflow import keras
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime

# so you can test different architectures
class HiddenLayer:
  def __init__(self, M1, M2, f=tf.nn.tanh, use_bias=True):
    self.W = tf.Variable(tf.random.normal(shape=(M1, M2)))
    self.params = [self.W]
    self.use_bias = use_bias
    if use_bias:
      self.b = tf.Variable(np.zeros(M2).astype(np.float32))
      self.params.append(self.b)
    self.f = f

  @tf.function
  def forward(self, X):
    if self.use_bias:
      a = tf.matmul(X, self.W) + self.b
    else:
      a = tf.matmul(X, self.W)
    return self.f(a)

# approximates pi(a | s)
class DQN:
  # D = Dimension of Feature Vec
  # K = Number of Outputs
  def __init__(self, D, K, hidden_layer_sizes, gamma, max_experiences=10000, min_experiences=100, batch_sz=32):
    self.K = K

    # create the graph
    self.layers = []
    M1 = D
    for M2 in hidden_layer_sizes:
      layer = HiddenLayer(M1, M2)
      self.layers.append(layer)
      M1 = M2

    # final layer
    layer = HiddenLayer(M1, K, lambda x: x)
    self.layers.append(layer)

    # add layers to keras model
    self.model = keras.Sequential()
    for layer in self.layers:
      self.model.add(keras.layers.Lambda(layer.forward))
    # define optimizer
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

    # create replay memory
    self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
    self.max_experiences = max_experiences
    self.min_experiences = min_experiences
    self.batch_sz = batch_sz
    self.gamma = gamma

  def set_session(self, session):
    self.session = session

  def copy_from(self, other):
    self.model.set_weights(other.model.get_weights())

  def predict(self, X):
    return self.model(np.atleast_2d(X))

  def train(self, target_network):
    # sample a random batch from buffer, do an iteration of GD
    if len(self.experience['s']) < self.min_experiences:
      # don't do anything if we don't have enough experience
      return

    # randomly select a batch
    idx = np.random.choice(len(self.experience['s']), size=self.batch_sz, replace=False)
    states = [self.experience['s'][i] for i in idx]
    actions = [self.experience['a'][i] for i in idx]
    rewards = [self.experience['r'][i] for i in idx]
    next_states = [self.experience['s2'][i] for i in idx]
    dones = [self.experience['done'][i] for i in idx]
    next_Q = np.max(target_network.predict(next_states), axis=1)
    targets = [r + self.gamma*next_q if not done else r for r, next_q, done in zip(rewards, next_Q, dones)]

    with tf.GradientTape() as t:
      selected_action_values = tf.reduce_sum(
        self.predict(states) * keras.backend.one_hot(actions, self.K), axis=1
        )
      cost = tf.reduce_sum(tf.math.square(targets - selected_action_values))
    grads = t.gradient(cost, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

  def add_experience(self, s, a, r, s2, done):
    if len(self.experience['s']) >= self.max_experiences:
      self.experience['s'].pop(0)
      self.experience['a'].pop(0)
      self.experience['r'].pop(0)
      self.experience['s2'].pop(0)
      self.experience['done'].pop(0)
    self.experience['s'].append(s)
    self.experience['a'].append(a)
    self.experience['r'].append(r)
    self.experience['s2'].append(s2)
    self.experience['done'].append(done)

  def sample_action(self, x, eps):
    # e-greedy
    if np.random.random() < eps:
      return np.random.choice(self.K)
    else:
      X = np.atleast_2d(x)
      return np.argmax(self.predict(X)[0])

def play_one(env, model, tmodel, eps, gamma, copy_period):
  observation = env.reset()
  done = False
  totalreward = 0
  iters = 0
  while not done and iters <= 1000:
    # if we reach 2000, just quit, don't want this going forever
    # the 200 limit seems a bit early
    action = model.sample_action(observation, eps)
    prev_observation = observation
    observation, reward, done, info = env.step(action)

    totalreward += reward
    if done:
      reward = -200

    # update the model
    model.add_experience(prev_observation, action, reward, observation, done)
    model.train(tmodel)

    iters += 1

    if iters % copy_period == 0:
      tmodel.copy_from(model)

  return totalreward

def main():
  env = gym.make('CartPole-v0')
  gamma = 0.99
  copy_period = 50

  D = len(env.observation_space.sample())
  K = env.action_space.n
  sizes = [200,200]
  model = DQN(D, K, sizes, gamma)
  tmodel = DQN(D, K, sizes, gamma)

  record=False
  if 'monitor' in sys.argv:
    record=True

  save_folder = config.getSaveFolder(os.path.basename(os.path.dirname(__file__)), os.path.basename(__file__).split('.')[0])
  if record is True:
    env = wrappers.Monitor(env, os.path.join(save_folder, 'monitor'))

  N = 500
  totalrewards = np.empty(N)
  for n in range(N):
    eps = 1.0/np.sqrt(n+1)
    totalreward = play_one(env, model, tmodel, eps, gamma, copy_period)
    totalrewards[n] = totalreward
    if n % 100 == 0:
      print("Episode: {}, Total Reward: {}, Average Reward (100): {}".format(n, totalreward, totalrewards[max(0, n-100):(n+1)].mean()))
  print("Average reward for last 100 episodes:", totalrewards[-100:].mean())
  print("Total steps:", -totalrewards.sum())

  config.plot_total_reward(totalrewards, save_folder, record)
  config.plot_running_avg(totalrewards, save_folder, record)


if __name__ == '__main__':
  main()

