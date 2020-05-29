import gym
import os
import sys
import config

from gym import wrappers
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))

def to_bin(value, bins):
    """ Return the indicies of the bins to which each value belongs """
    return np.digitize(x=[value], bins=bins)[0]

class FeatureTransformer:
    def __init__(self, numBins):
        # These bins have been hard-coded, replace with a dynamic bin-size
        # assessor that runs once then saves the bin-sizes.
        self.numBins = numBins
        self.cart_position_bins = np.linspace(-2.4, 2.4, numBins)
        self.cart_velocity_bins = np.linspace(-2, 2, numBins)
        self.pole_angle_bins = np.linspace(-0.4, 0.4, numBins)
        self.pole_velocity_bins = np.linspace(-3.5, 3.5, numBins)
    
    def transform(self, observation):
        cart_pos, cart_vel, pole_angle, pole_vel = observation
        return build_state([
            to_bin(cart_pos, self.cart_position_bins),
            to_bin(cart_vel, self.cart_velocity_bins),
            to_bin(pole_angle, self.pole_angle_bins),
            to_bin(pole_vel, self.pole_velocity_bins)
        ])

class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.feature_transformer = feature_transformer

        num_states = (feature_transformer.numBins+1)**env.observation_space.shape[0]
        num_actions = env.action_space.n
        # Initialise Q table
        self.Q = np.random.uniform(low=-1, high=1, size=(num_states, num_actions))

    def predict(self, s):
        x = self.feature_transformer.transform(s)
        return self.Q[x]

    def update(self, s, a, G):
        """ s = state, a = action, G = target return """
        x = self.feature_transformer.transform(s)
        self.Q[x,a] += 10e-3*(G - self.Q[x,a])

    def sample_action(self, s, eps):
        # Epislon Greedy
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            p = self.predict(s)
            return np.argmax(p)

def play_one(model, eps, gamma):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0

    while not done and iters < 10000:
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, _ = env.step(action)
        totalreward += reward

        # Modifying the base reward structure, apparently
        # this can be controversial. Here we punish
        # failing the experiment early.
        if done and iters < 199:
            reward -= 300

        # G is the immediate reward + the maximum reward we expected
        # to recieve at the previous step, this is the Q-Learning definition
        G = reward + gamma*np.max(model.predict(observation))
        model.update(prev_observation, action, G)
        iters += 1

    return totalreward

def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()

if __name__ == '__main__':
    print("\nCartPole Problem: Q Learning Bins\n")
    env = gym.make('CartPoleLong-v0')
    ft = FeatureTransformer(numBins=9)
    model = Model(env, ft)
    gamma = 0.9

    save_folder = config.getSaveFolder('cartpole', os.path.basename(__file__).split('.')[0])
    
    # Save if monitor is called in cmdline arguments
    if 'monitor' in sys.argv:
        env = wrappers.Monitor(env, os.path.join(save_folder, 'monitor'))

    N = 10000
    totalrewards = np.empty(N)
    for n in range(N):
        # Decay epislon over time
        eps = 1.0/np.sqrt(n+1)
        totalreward = play_one(model, eps, gamma)
        totalrewards[n] = totalreward
        if n%100 == 0:
            print("Episode: {}, Total Reward: {}, Eps: {}".format(n, totalreward, eps))
    print("Average Reward for last 100 episodes: {}".format(totalrewards[-100:].mean()))
    print("Total Steps: {}".format(totalrewards.sum()))
    
    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.xlabel("Iterations")
    plt.ylabel("Reward")
    plt.show()
    if 'monitor' in sys.argv:
        plt.savefig(os.path.join(save_folder, 'random_search-episode_lengths.png'), dpi=600)
    plot_running_avg(totalrewards)
