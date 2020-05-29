import gym
import os
import sys
import config

import numpy as np
import matplotlib.pyplot as plt 
from gym import wrappers

from mountaincart.q_learning_rbf import FeatureTransformer, plot_cost_to_go

# Custom Base Model
class BaseModel:
  def __init__(self, D):
    # Xavier Initialisation of Weights
    self.w = np.random.randn(D) / np.sqrt(D)

  def partial_fit(self, state, target, eligibility, lr=10e-3):
    self.w += lr*(target - state.dot(self.w)*eligibility)

  def predict(self, X):
    X = np.array(X)
    return X.dot(self.w)

# Holds one BaseModel for each action
class Model:
  def __init__(self, env, feature_transformer):
    self.env = env
    self.models = []
    self.feature_transformer = feature_transformer

    D = feature_transformer.dimensions
    # Eligibility trace has size equal to action_space * feature dimensions
    self.eligibilities = np.zeros((env.action_space.n, D))
    for i in range(env.action_space.n):
      model = BaseModel(D)
      self.models.append(model)

  def predict(self, s):
    X = self.feature_transformer.transform([s])
    assert(len(X.shape) == 2)
    result = np.stack([m.predict(X) for m in self.models]).T
    assert(len(result.shape) == 2)
    return result

  def update(self, s, a, G, gamma, lambda_):
    X = self.feature_transformer.transform([s])
    assert(len(X.shape) == 2)
    # Eligibility[t] = Grad[Theta]*V(S_t) + Gamma * Lambda * Eligibility[t-1]
    # Eligibility trace update formulae
    # Lambda -> How much of the past we want to keep i.e. Td(Lambda)
    # Gamma  -> Discount rate
    self.eligibilities *= gamma*lambda_
    # Action activates the relevant eligibility trace for update
    # and adds V[S_t] which is just our state feature vector
    self.eligibilities[a] += X[0]
    # We then use the eligbility trace, G and state vector X[0] to update
    # our models that we use to predict our actions
    self.models[a].partial_fit(X[0], G, self.eligibilities[a])

  def sample_action(self, s, eps):
    if np.random.random() < eps:
      return self.env.action_space.sample()
    else:
      return np.argmax(self.predict(s))

# lambda is reserved variable name
def play_one(model, eps, gamma, lambda_):
  env = gym.make('MountainCar-v0')
  observation = env.reset()
  done = False
  totalreward = 0
  iters = 0

  while not done:
    # sample action
    action = model.sample_action(observation, eps)
    prev_observation = observation
    observation, reward, done, _ = env.step(action)
    next = model.predict(observation)
    assert(next.shape == (1, env.action_space.n))
    G = reward + gamma*np.max(model.predict(observation)[0])
    model.update(prev_observation, action, G, gamma, lambda_)
    totalreward += reward
    iters +=1 

  return totalreward 

def main():
  # replace regressor
  env = gym.make('MountainCar-v0')
  ft = FeatureTransformer(env)
  # Use local td-lambda model
  model = Model(env, ft)
  gamma = 0.99
  lambda_ = 0.9

  record = False
  save_folder = config.getSaveFolder('cartpole', os.path.basename(__file__).split('.')[0])
  if 'monitor' in sys.argv:
    record = True
    env = wrappers.Monitor(env, os.path.join(save_folder, 'monitor'))

  N = 300
  totalrewards = np.empty(N)
  costs = np.empty(N)
  for n in range(N):
    eps = 0.1*(0.97**n)
    totalreward = play_one(model, eps, gamma, lambda_)
    totalrewards[n] = totalreward
    print("Episode: {}, Total Reward: {}".format(n, totalreward))
  print("Average reward for last 100 episodes:", totalrewards[-100:].mean())
  print("Total steps:", -totalrewards.sum())

  config.plot_total_reward(totalrewards, save_folder, record)
  config.plot_running_avg(totalrewards, save_folder, record)
  plot_cost_to_go(env, model, save_folder)
  
if __name__ == '__main__':
  main()
