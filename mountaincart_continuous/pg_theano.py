import gym
import os
import sys
import config
import numpy as np 

import theano
import theano.tensor as T

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

# helper for adam optimizer
# use tensorflow defaults
# https://arxiv.org/abs/1412.6980
def adam(cost, params, lr0=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
  grads = T.grad(cost, params)
  updates = []
  time = theano.shared(0)
  new_time = time + 1
  updates.append((time, new_time))
  lr = lr0*T.sqrt(1 - beta2**new_time) / (1 - beta1**new_time)
  for p, g in zip(params, grads):
    m = theano.shared(p.get_value() * 0.)
    v = theano.shared(p.get_value() * 0.)
    new_m = beta1*m + (1 - beta1)*g
    new_v = beta2*v + (1 - beta2)*g*g
    new_p = p - lr*new_m / (T.sqrt(new_v) + eps)
    updates.append((m, new_m))
    updates.append((v, new_v))
    updates.append((p, new_p))
  return updates


class HiddenLayer:
  def __init__(self, M1, M2, f=T.nnet.relu, use_bias=True, zeros=False):
    if zeros:
      W = np.zeros((M1, M2))
    else:
      W = np.random.randn(M1, M2) * np.sqrt(2. / M1)

    self.W = theano.shared(W)
    self.params = [self.W]
    self.use_bias = use_bias
    if use_bias:
      self.b = theano.shared(np.zeros(M2))
      self.params += [self.b]
    self.f = f

  def forward(self, X):
    a = X.dot(self.W)
    if self.use_bias:
      a+= self.b
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

    # final layer mean
    self.mean_layer = HiddenLayer(M1, 1, lambda x: x, use_bias=False, zeros=True)

    # final layer variance
    self.var_layer = HiddenLayer(M1, 1, T.nnet.softplus, use_bias=False, zeros=False)

    # get all params for gradient later
    params = self.mean_layer.params + self.var_layer.params
    for layer in self.hidden_layers:
      params += layer.params

    # inputs and targets
    X = T.matrix('X')
    actions = T.vector('actions')
    advantages = T.vector('advantages')
    # target_value = T.vector('target_value')

    # get final hidden layer
    Z = X
    for layer in self.hidden_layers:
      Z = layer.forward(Z)

    mean = self.mean_layer.forward(Z).flatten()
    var = self.var_layer.forward(Z).flatten() + 1e-5 # smoothing

    # Define function for log pdf of a gaussian
    def log_pdf(actions, mean, var):
      k1 = T.log(2*np.pi*var)
      k2 = (actions - mean)**2 / var
      return -0.5*(k1 + k2)

    # Entropy of a gaussian
    def entropy(var):
      return 0.5*T.log(2*np.pi*np.e*var)

    log_probs = log_pdf(actions, mean, var)
    cost = -T.sum(advantages * log_probs + 0.1*entropy(var))
    updates = adam(cost, params)

    # compile functions
    self.train_op = theano.function(
      inputs=[X, actions, advantages],
      updates=updates,
      allow_input_downcast=True
    )

    # alternatively, we could create a RandomStream and sample from
    # the Gaussian using Theano code
    self.predict_op = theano.function(
      inputs=[X],
      outputs=[mean, var],
      allow_input_downcast=True
    )

  def partial_fit(self, X, actions, advantages):
    X = np.atleast_2d(X)
    X = self.ft.transform(X)
    actions = np.atleast_1d(actions)
    advantages = np.atleast_1d(advantages)
    self.train_op(X, actions, advantages)

  def predict(self, X):
    X = np.atleast_2d(X)
    X = self.ft.transform(X)
    return self.predict_op(X)

  def sample_action(self, X):
    pred = self.predict(X)
    mu = pred[0][0]
    v = pred[1][0]
    a = np.random.randn()*np.sqrt(v) + mu
    # Environment limitations
    return min(max(a, -1), 1)

# approximates V(s)
class ValueModel:
  def __init__(self, D, ft, hidden_layer_sizes=[]):
    self.ft = ft

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

    # get all params for gradient later
    params = []
    for layer in self.layers:
      params += layer.params

    # inputs and targets
    X = T.matrix('X')
    Y = T.vector('Y')

    # calculate output and cost
    Z = X
    for layer in self.layers:
      Z = layer.forward(Z)
    Y_hat = T.flatten(Z)
    cost = T.sum((Y - Y_hat)**2)

    # specify update rule
    updates = adam(cost, params, lr0=1e-2)

    # compile functions
    self.train_op = theano.function(
      inputs=[X, Y],
      updates=updates,
      allow_input_downcast=True
    )
    self.predict_op = theano.function(
      inputs=[X],
      outputs=Y_hat,
      allow_input_downcast=True
    )

  def partial_fit(self, X, Y):
    X = np.atleast_2d(X)
    X = self.ft.transform(X)
    Y = np.atleast_1d(Y)
    self.train_op(X, Y)

  def predict(self, X):
    X = np.atleast_2d(X)
    X = self.ft.transform(X)
    return self.predict_op(X)


def play_one_td(env, pmodel, vmodel, gamma, train=True):
  observation = env.reset()
  done = False
  totalreward = 0
  iters = 0

  while not done and iters < 10000:
    # if we reach 2000, just quit, don't want this going forever
    # the 200 limit seems a bit early
    action = pmodel.sample_action(observation)
    prev_observation = observation
    observation, reward, done, _ = env.step([action])

    totalreward += reward

    # update the models
    if train:
      V_next = vmodel.predict(observation)
      G = reward + gamma*V_next
      advantage = G - vmodel.predict(prev_observation)
      pmodel.partial_fit(prev_observation, action, advantage)
      vmodel.partial_fit(prev_observation, G)

    iters += 1

  return totalreward

def main():
  env = gym.make('MountainCarContinuous-v0')
  ft = FeatureTransformer(env, n_components=100)
  D = ft.dimensions
  pmodel = PolicyModel(D, ft)
  vmodel = ValueModel(D, ft)
  gamma = 0.99

  record = False
  if 'monitor' in sys.argv:
    record=True

  save_folder = config.getSaveFolder(os.path.basename(os.path.dirname(__file__)), os.path.basename(__file__).split('.')[0])
  if record is True:
    env = wrappers.Monitor(env, os.path.join(save_folder, 'monitor'))

  N = 50
  totalrewards = np.empty(N)
  for n in range(N):
    totalreward = play_one_td(env, pmodel, vmodel, gamma)
    totalrewards[n] = totalreward
    if n % 1 == 0:
      print("Episode: {}, Total Reward: {:.2f}".format(n, totalreward))
  print("Average reward over 100 episodes with best models: {}".format(totalrewards[-100:].mean()))

  config.plot_total_reward(np.array(totalrewards), save_folder, record)
  config.plot_running_avg(np.array(totalrewards), save_folder, record)
  plot_cost_to_go(env, vmodel, save_folder, record)


if __name__ == '__main__':
  main()