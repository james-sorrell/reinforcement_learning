# https://deeplearningcourses.com/c/cutting-edge-artificial-intelligence
import time
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os


class Agent:
    def __init__(self, Network, ob_space, ac_space, nenvs, nsteps, nstack,
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
                 alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6)):

        tf.config.threading.set_intra_op_parallelism_threads=nenvs
        tf.config.threading.set_inter_op_parallelism_threads=nenvs

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

        #self.step_model = Network(ob_space, ac_space, nenvs, 1, nstack, lr, alpha, epsilon, ent_coef, vf_coef, max_grad_norm)
        self.model = Network(ob_space, ac_space, nenvs, nsteps, nstack, lr, alpha, epsilon, ent_coef, vf_coef, max_grad_norm)
        self.step = self.model.step
        self.value = self.model.value

    def train(self, states, rewards, actions, values):
        advantages = rewards - values
        policy_loss, value_loss, policy_entropy = self.model.train(states, actions, advantages, rewards)
        return policy_loss, value_loss, policy_entropy

    def save(self, save_path):
        self.model.network.save(save_path)

    def load(self, load_path):
        self.model.network = keras.models.load_model(load_path)


class Runner:
    def __init__(self, env, agent, nsteps=5, nstack=4, gamma=0.99):
        self.env = env
        self.agent = agent
        nh, nw, nc = env.observation_space.shape
        nenv = env.num_envs
        self.batch_ob_shape = (nenv * nsteps, nh, nw, nc * nstack)
        self.state = np.zeros((nenv, nh, nw, nc * nstack), dtype=np.uint8)
        self.nc = nc
        obs = env.reset()
        self.update_state(obs)
        self.gamma = gamma
        self.nsteps = nsteps
        self.dones = [False for _ in range(nenv)]
        self.total_rewards = [] # store all workers' total rewards
        self.real_total_rewards = []

    def update_state(self, obs):
        # Do frame-stacking here instead of the FrameStack wrapper to reduce IPC overhead
        self.state = np.roll(self.state, shift=-self.nc, axis=3)
        self.state[:, :, :, -self.nc:] = obs

    def discount_with_dones(self, rewards, dones, gamma):
        discounted = []
        r = 0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            # (1 - done means terminal states  are defined purely by reward)
            r = reward + gamma * r * (1. - done)
            discounted.append(r)
        return discounted[::-1]

    def run(self):
        mb_states, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
        for n in range(self.nsteps):
            actions, values = self.agent.step(self.state)
            mb_states.append(np.copy(self.state))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, infos = self.env.step(actions)
            for done, info in zip(dones, infos):
                if done:
                    self.total_rewards.append(info['reward'])
                    if info['total_reward'] != -1:
                        self.real_total_rewards.append(info['total_reward'])
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.state[n] = self.state[n] * 0
            self.update_state(obs)
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        # batch of steps to batch of rollouts
        mb_states = np.asarray(mb_states, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_dones = mb_dones[:, 1:]
        last_values = self.agent.value(self.state).numpy()
        # discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = self.discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
            else:
                rewards = self.discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        return mb_states, mb_rewards, mb_actions, mb_values


def set_global_seeds(i):
    tf.random.set_seed(i)
    np.random.seed(i)

def learn(network, env, seed, new_session=True,  nsteps=5, nstack=4, total_timesteps=int(80e6),
          vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4,
          epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=1000):
    
    #tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    env_id = env.env_id
    save_name = os.path.join('models', env_id + '.save')
    ob_space = env.observation_space
    ac_space = env.action_space
    agent = Agent(Network=network, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs,
                  nsteps=nsteps, nstack=nstack,
                  ent_coef=ent_coef, vf_coef=vf_coef,
                  max_grad_norm=max_grad_norm,
                  lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps)
    if os.path.exists(save_name):
        agent.load(save_name)

    runner = Runner(env, agent, nsteps=nsteps, nstack=nstack, gamma=gamma)

    nbatch = nenvs * nsteps
    tstart = time.time()
    for update in range(1, total_timesteps // nbatch + 1):
        states, rewards, actions, values = runner.run()
        policy_loss, value_loss, policy_entropy = agent.train(
            states, rewards, actions, values)
        nseconds = time.time() - tstart
        fps = int((update * nbatch) / nseconds)
        if update % log_interval == 0 or update == 1:
            print(' - - - - - - - ')
            print("nupdates", update)
            print("total_timesteps", update * nbatch)
            print("fps", fps)
            print("policy_entropy", float(policy_entropy))
            print("value_loss", float(value_loss))

            # total reward
            r = runner.total_rewards[-100:] # get last 100
            tr = runner.real_total_rewards[-100:]
            if len(r) == 100:
                print("avg reward (last 100):", np.mean(r))
            if len(tr) == 100:
                print("avg total reward (last 100):", np.mean(tr))
                print("max (last 100):", np.max(tr))

            agent.save(save_name)

    env.close()
    agent.save(save_name)
