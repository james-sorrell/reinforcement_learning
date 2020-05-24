# -*- coding: utf-8 -*-
import gym
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
    testing_length = 100
    permutation_count = 100
    for _ in range(permutation_count):
        new_params = np.random.random(4)*2 - 1
        # 100 is number of times each param set is tested
        avg_length = play_multiple_epsiodes(env, testing_length, new_params)
        episode_lengths.append(avg_length)
        if avg_length > best:
            params = new_params
            best = avg_length
    return episode_lengths, params


gym.envs.register(
    id='CartPoleLong-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 10000},
    reward_threshold=195.0,
)

env = gym.make('CartPoleLong-v0')
episode_lengths, params = random_search(env)
plt.plot(episode_lengths)
plt.show()

print("Visualised run with winning parameters: {}".format(params))
play_multiple_epsiodes(env, 100, params, True)