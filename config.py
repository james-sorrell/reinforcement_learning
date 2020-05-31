import gym
import os

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

DATA_FOLDER = "data"

def create_folder(path):
    try:
        # Create target Directory
        os.mkdir(path)
        print("Directory " , path ,  " Created ") 
    except FileExistsError:
        print("Directory " , path ,  " already exists")

def dateStr():
    return datetime.today().strftime('%Y-%m-%d-%H-%M-%S')

def getSaveFolder(experiment_category, experiment_name):
    return os.path.join(DATA_FOLDER, experiment_category, experiment_name, dateStr())

def plot_running_avg(totalrewards, save_folder, save_fig=True):
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
  plt.plot(running_avg)
  plt.title("Running Average")
  plt.xlabel("Iteration")
  plt.ylabel("Reward")
  if save_fig:
    plt.savefig(os.path.join(save_folder, 'running_average.png'))
  plt.show()

def plot_total_reward(total_rewards, save_folder, record=False):
  plt.plot(total_rewards)
  plt.title("Total Reward vs Iteration")
  plt.ylabel("Reward")
  plt.xlabel("Iteration")
  if record is True:
      plt.savefig(os.path.join(save_folder, 'total_rewards.png'))
  plt.show()

gym.envs.register(
    id='CartPoleLong-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps= 1000,
    reward_threshold=195.0,
)