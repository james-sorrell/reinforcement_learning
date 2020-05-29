import gym
import os

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

def plot_running_avg(totalrewards, save_folder):
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
  plt.plot(running_avg)
  plt.title("Running Average")
  plt.xlabel("Iteration")
  plt.ylabel("Reward")
  if 'monitor' in sys.argv:
    plt.savefig(os.path.join(save_folder, 'running_average.png'))
  plt.show()

gym.envs.register(
    id='CartPoleLong-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
    reward_threshold=195.0,
)