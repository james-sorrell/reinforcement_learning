import gym
import os
import sys

import config

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gym import wrappers
from datetime import datetime
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler

class SGDRegressor:
    def __init__(self, D, learning_rate=10e-2):
        self.w = np.random.randn(D) / np.sqrt(D)
        self.lr = learning_rate
    
    def partial_fit(self, X, Y):
        self.w += self.lr*(Y-X.dot(self.w)).dot(X)
        
    def predict(self, X):
        return X.dot(self.w)

class FeatureTransformer:
  def __init__(self, env, n_components=500):
    # Uniform sampling around points that we think we will see during training
    observation_examples = np.random.random((20000,4))*2-2
    scaler = StandardScaler()
    scaler.fit(observation_examples)

    # Used to convert a state to a featurizes represenation.
    # We use RBF kernels with different variances to cover different parts of the space
    featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=n_components))
            ])
    example_features = featurizer.fit_transform(scaler.transform(observation_examples))

    self.dimensions = example_features.shape[1]
    self.scaler = scaler
    self.featurizer = featurizer

  def transform(self, observations):
    # Scaler removes mean and scales to unit variance
    scaled = self.scaler.transform(observations)
    return self.featurizer.transform(scaled)


# Holds one SGDRegressor for each action
class Model:
  def __init__(self, env, feature_transformer, learning_rate):
    self.env = env
    self.models = []
    self.feature_transformer = feature_transformer
    for _ in range(env.action_space.n):
      model = SGDRegressor(feature_transformer.dimensions, learning_rate=learning_rate)
      self.models.append(model)

  def predict(self, s):
    X = self.feature_transformer.transform(np.atleast_2d(s))
    # For each of our models, (one per action), attain reward estimate
    return np.stack([m.predict(X) for m in self.models]).T

  def update(self, s, a, G):
    X = self.feature_transformer.transform(np.atleast_2d(s))
    # We only update the model that corresponds to the action we took
    self.models[a].partial_fit(X, [G])

  def sample_action(self, s, eps):
    if np.random.random() < eps:
      return self.env.action_space.sample()
    else:
      return np.argmax(self.predict(s))

# returns a list of states_and_rewards, and the total reward
def play_one(model, env, eps, gamma):
  observation = env.reset()
  done = False
  totalreward = 0
  iters = 0
  while not done:
    action = model.sample_action(observation, eps)
    prev_observation = observation
    observation, reward, done, _ = env.step(action)

    # Modifying the base reward structure, apparently
    # this can be controversial. Here we punish
    # failing the experiment early.
    if done and iters < 199:
        reward -= 300

    # update the model
    next = model.predict(observation)
    G = reward + gamma*np.max(next)
    model.update(prev_observation, action, G)

    if reward == 1:
        totalreward += reward
    iters += 1

  return totalreward


def main():
  env = gym.make('CartPoleLong-v0')

  if 'monitor' in sys.argv:
    record=True

  ft = FeatureTransformer(env)
  # Learning rate set to 10e-2
  model = Model(env, ft, 10e-2)
  gamma = 0.99

  save_folder = config.getSaveFolder('cartpole', os.path.basename(__file__).split('.')[0])
  if record is True:
    env = wrappers.Monitor(env, os.path.join(save_folder, 'monitor'))

  N = 500
  totalrewards = np.empty(N)
  for n in range(N):
    eps = 1.0/np.sqrt(n+1)
    totalreward = play_one(model, env, eps, gamma)
    totalrewards[n] = totalreward
    if n % 100 == 0:
      print("Episode: {}, Total Reward: {}".format(n, totalreward))
  print("Average reward for last 100 episodes:", totalrewards[-100:].mean())
  print("Total steps:", -totalrewards.sum())

  config.plot_total_reward(totalrewards, save_folder, record)
  config.plot_running_avg(totalrewards, save_folder, record)


if __name__ == '__main__':
  main()