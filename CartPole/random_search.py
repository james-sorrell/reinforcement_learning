# -*- coding: utf-8 -*-
import os
import gym

import config

from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt

def get_action(s, w):
    return 1 if s.dot(w) > 0 else 0

def play_one_episode(env, params, render_env=False):
    """ Play the episode once for a given set of parameters
    and return how long it lasted """
    observation = env.reset()
    done = False
    t = 0
    
    # Limit actions to 10000
    while not done and t < 10000:
        if render_env is True:
            env.render()
        t += 1
        action = get_action(observation, params)
        observation, _, done, _ = env.step(action)
        if done:
            break
    
    return t

def play_multiple_epsiodes(env, T, params, render_env=False):
    """ For a given set of parameters, play the episode T times
    and average the resulting number of steps the environment
    lasted."""
    episode_lengths = np.empty(T)
    for i in range(T):
        episode_lengths[i] = play_one_episode(env, params, render_env)
    avg_length = episode_lengths.mean()
    return avg_length

def random_search(env):
    episode_lengths = []
    best = 0
    params = None
    tests_per_permutation = 100
    number_of_permutations = 100
    for _ in range(number_of_permutations):
        new_params = np.random.random(4)*2 - 1
        # 100 is number of times each param set is tested
        avg_length = play_multiple_epsiodes(env, tests_per_permutation, new_params)
        episode_lengths.append(avg_length)
        if avg_length > best:
            print("New Best: {} turns".format(best))
            params = new_params
            best = avg_length
    return episode_lengths, params

if __name__ == '__main__':
    print("\nCartPole Problem: Random Search\n")
    env = gym.make('CartPoleLong-v0')
    episode_lengths, params = random_search(env)

    save_folder = config.getSaveFolder('cartpole', os.path.basename(__file__).split('.')[0])
    plt.plot(episode_lengths)
    plt.title("Random Search: Episode Lengths")
    plt.ylabel('Episode Lengths')
    plt.xlabel('Permutation Number')
    plt.show(block=False)
    env = wrappers.Monitor(env, os.path.join(save_folder, 'monitor'), force=True)
    plt.savefig(os.path.join(save_folder, 'random_search-episode_lengths.png'), dpi=600)
    print("Visualising run with winning parameters: {}".format(params))
    play_one_episode(env, params)