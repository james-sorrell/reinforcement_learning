import gym
import os
import sys
import matplotlib

import config

import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
import mountaincart.q_learning_rbf as q_learning

# Custom SGD Regressor
class SGDRegressor:
  def __init__(self, **kwargs):
    print("Using n_step SGD Regressor")
    self.w = None
    self.lr = 1e-2

  # Weights initialised in partial fit
  def partial_fit(self, X, Y):
    if self.w is None:
      D = X.shape[1]
      self.w = np.random.randn(D) / np.sqrt(D)
    self.w += self.lr*(Y - X.dot(self.w)).dot(X)

  def predict(self, X):
    return X.dot(self.w)


# calculate everything up to max[Q(s,a)]
# Ex.
# R(t) + gamma*R(t+1) + ... + (gamma^(n-1))*R(t+n-1) + (gamma^n)*max[Q(s(t+n), a(t+n))]
# def calculate_return_before_prediction(rewards, gamma):
#   ret = 0
#   for r in reversed(rewards[1:]):
#     ret += r + gamma*ret
#   ret += rewards[0]
#   return ret
def play_one(model, eps, gamma, n=5):
  observation = env.reset()
  done = False
  totalreward = 0
  rewards = []
  states = []
  actions = []
  iters = 0
  # array of multipliers for reward decay calculation
  multiplier = np.array([gamma]*n)**np.arange(n)
  while not done:
    # sample action
    action = model.sample_action(observation, eps)
    # track n states/actions/rewards
    states.append(observation)
    actions.append(action)
    observation, reward, done, _ = env.step(action)

    rewards.append(reward)

    if len(rewards) >= n:
      # discount*reward + discount^2*reward(t-1) etc
      return_up_to_prediction = multiplier.dot(rewards[-n:])
      G = return_up_to_prediction + (gamma**n)*np.max(model.predict(observation)[0])
      model.update(states[-n], actions[-n], G)

    totalreward += reward
    iters +=1 

  # reset cache
  if n == 1:
    rewards = []
    states = []
    actions = []
  else:
    rewards = rewards[-n+1:]
    states = states[-n+1:]
    actions = actions[-n+1:]

  # goal was achieved, as per documentation
  if observation[0] >= 0.5:
    # we actually made it to our goal
    while len(rewards) > 0:
      # discount*reward + discount^2*reward(t-1) etc as much as we have left
      G = multiplier[:len(rewards)].dot(rewards)
      model.update(states[0], actions[0], G)
      rewards.pop(0)
      states.pop(0)
      actions.pop(0)
    else:
      # we did not make it to the goal
      while len(rewards) > 0:
        guess_rewards = rewards + [-1]*(n-len(rewards))
        G = multiplier.dot(guess_rewards)
        model.update(states[0], actions[0], G)
        rewards.pop(0)
        states.pop(0)
        actions.pop(0)

  return totalreward


if __name__ == '__main__':
  # replace regressor
  q_learning.SGDRegressor = SGDRegressor
  env = gym.make('MountainCar-v0')
  ft = q_learning.FeatureTransformer(env)
  model = q_learning.Model(env, ft, "constant")
  gamma = 0.99

  record = False
  save_folder = config.getSaveFolder(os.path.basename(os.path.dirname(__file__)), os.path.basename(__file__).split('.')[0])
  if 'monitor' in sys.argv:
    record = True
    env = wrappers.Monitor(env, os.path.join(save_folder, 'monitor'))

  N = 300
  totalrewards = np.empty(N)
  costs = np.empty(N)
  for n in range(N):
    eps = 0.1*(0.97**n)
    totalreward = play_one(model, eps, gamma)
    totalrewards[n] = totalreward
    print("Episode: {}, Total Reward: {}".format(n, totalreward))
  print("Average reward for last 100 episodes:", totalrewards[-100:].mean())
  print("Total steps:", -totalrewards.sum())

  config.plot_total_reward(totalrewards, save_folder, record)
  config.plot_running_avg(totalrewards, save_folder, record)
  q_learning.plot_cost_to_go(env, model, save_folder)