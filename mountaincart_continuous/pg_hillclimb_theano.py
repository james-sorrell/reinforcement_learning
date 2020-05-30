import gym
import os
import sys
import config
import numpy as np 
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
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
  def __init__(self, M1, M2, f=T.nnet.relu, use_bias=True, zeros=False):
    if zeros:
      W = np.zeros((M1, M2))
    else:
      W = np.random.randn(M1, M2) * np.sqrt(2 / M1)
    self.W = theano.shared(W)
    self.params = [self.W]
    self.use_bias = use_bias
    if use_bias:
      self.b = theano.shared(np.zeros(M2))
      self.params += [self.b]
    self.f = f

  def forward(self, X):
    if self.use_bias:
      a = X.dot(self.W) + self.b
    else:
      a = X.dot(self.W)
    return self.f(a)


class PolicyModel:
  def __init__(self, ft, D, hidden_layer_sizes_mean=[], hidden_layer_sizes_var=[]):
    # save inputs for copy
    self.ft = ft
    self.D = D
    self.hidden_layer_sizes_mean = hidden_layer_sizes_mean
    self.hidden_layer_sizes_var = hidden_layer_sizes_var
    
    ##### model the mean #####
    self.mean_layers = []
    M1 = D
    for M2 in hidden_layer_sizes_mean:
      layer = HiddenLayer(M1, M2)
      self.mean_layers.append(layer)
      M1 = M2

    # final layer
    layer = HiddenLayer(M1, 1, lambda x: x, use_bias=False, zeros=True)
    self.mean_layers.append(layer)

    ##### model the variance #####
    self.var_layers = []
    M1 = D
    for M2 in hidden_layer_sizes_var:
      layer = HiddenLayer(M1, M2)
      self.var_layers.append(layer)
      M1 = M2

    # final layer -> Variance needs to be greater than zero,
    # hence we use softplus as the activation.
    layer = HiddenLayer(M1, 1, T.nnet.softplus, use_bias=False, zeros=False)
    self.var_layers.append(layer)

    # get all params for gradient later
    params = []
    for layer in (self.mean_layers + self.var_layers):
      params += layer.params
    self.params = params

    # inputs and targets
    X = T.matrix('X')
    #actions = T.vector('actions')
    #advantages = T.vector('advantages')

    # calculate output and cost
    def get_output(layers):
      Z = X
      for layer in layers:
        Z = layer.forward(Z)
      return Z.flatten()

    mean = get_output(self.mean_layers)
    var = get_output(self.var_layers) + 1e-4 # smoothing

    # alternatively, we could create a RandomStream and sample from
    # the Gaussian using Theano code
    self.predict_op = theano.function(
      inputs=[X],
      outputs=[mean, var],
      allow_input_downcast=True
    )

  def predict(self, X):
    X = np.atleast_2d(X)
    X = self.ft.transform(X)
    return self.predict_op(X)

  def sample_action(self, X):
    pred = self.predict(X)
    mu = pred[0][0]
    v = pred[1][0]
    a = np.random.randn()*np.sqrt(v) + mu
    # Environment only allows actions between -1 and 1
    return min(max(a, -1), 1)

  # Hill Climbing Features 
  def copy(self):
    clone = PolicyModel(self.ft, self.D, self.hidden_layer_sizes_mean, self.hidden_layer_sizes_mean)
    clone.copy_from(self)
    return clone

  def copy_from(self, other):
    # self is being copied from other
    for p, q in zip(self.params, other.params):
      v = q.get_value()
      p.set_value(v)

  def perturb_params(self):
    # For each parameter, we have a probability that we
    # will keep the parameter and add some noise to it,
    # otherwise, we reset the parameter.
    for p in self.params:
      v = p.get_value()
      noise = np.random.randn(*v.shape) / np.sqrt(v.shape[0]) * 5.0
      if np.random.random() < 0.1:
        # with probability 0.1 start completely from scratch
        p.set_value(noise)
      else:
        p.set_value(v + noise)


def play_one(env, pmodel, gamma):
  observation = env.reset()
  done = False
  totalreward = 0
  iters = 0

  while not done and iters < 2000:
    # if we reach 2000, just quit, don't want this going forever
    # the 200 limit seems a bit early
    action = pmodel.sample_action(observation)
    # oddly, the mountain car environment requires the action to be in
    # an object where the actual action is stored in object[0]
    observation, reward, done, _ = env.step([action])
    totalreward += reward
    iters += 1
  return totalreward


def play_multiple_episodes(env, T, pmodel, gamma, print_iters=False):
  totalrewards = np.empty(T)

  for i in range(T):
    totalrewards[i] = play_one(env, pmodel, gamma)

    if print_iters:
      print("{} - Average so far: {}".format(i, totalrewards[:(i+1)].mean()))

  avg_totalrewards = totalrewards.mean()
  print("Average Total Reward: {}".format(avg_totalrewards))
  return avg_totalrewards

# Brains of the Hill Climbing Algorithm
def random_search(env, pmodel, gamma):
  totalrewards = []
  best_avg_totalreward = float('-inf')
  best_pmodel = pmodel
  num_episodes_per_param_test = 3

  for t in range(100):
    tmp_pmodel = best_pmodel.copy()
    tmp_pmodel.perturb_params()

    avg_totalrewards = play_multiple_episodes(
      env,
      num_episodes_per_param_test,
      tmp_pmodel,
      gamma
    )
    totalrewards.append(avg_totalrewards)

    if avg_totalrewards > best_avg_totalreward:
      best_pmodel = tmp_pmodel
      best_avg_totalreward = avg_totalrewards
  return totalrewards, best_pmodel

def main():
  env = gym.make('MountainCarContinuous-v0')
  ft = FeatureTransformer(env, n_components=100)
  D = ft.dimensions
  pmodel = PolicyModel(ft, D, [], [])
  gamma = 0.99

  record = False
  if 'monitor' in sys.argv:
    record=True

  save_folder = config.getSaveFolder(os.path.basename(os.path.dirname(__file__)), os.path.basename(__file__).split('.')[0])
  if record is True:
    env = wrappers.Monitor(env, os.path.join(save_folder, 'monitor'))


  totalrewards, pmodel = random_search(env, pmodel, gamma)
  print("Max Reward: {}".format(np.max(totalrewards)))
  # play 100 episodes and check the average
  avg_totalrewards = play_multiple_episodes(env, 100, pmodel, gamma, print_iters=True)
  print("Average reward over 100 episodes with best models: {}".format(avg_totalrewards))

  config.plot_total_reward(np.array(totalrewards), save_folder, record)
  config.plot_running_avg(np.array(totalrewards), save_folder, record)
  # plot_cost_to_go(env, model, save_folder)



if __name__ == '__main__':
  main()
