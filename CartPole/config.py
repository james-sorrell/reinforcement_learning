from datetime import datetime
import gym

DATA_FOLDER = "Data"

def dateStr():
    return datetime.today().strftime('%Y-%m-%d-%H-%M-%S')

gym.envs.register(
    id='CartPoleLong-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 30000},
    reward_threshold=195.0,
)