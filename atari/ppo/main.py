import argparse
import gym
import random
import tensorflow as tf
import numpy as np
from tqdm import trange
from tensorflow import keras
from network import SharedModel
from subproc_env import EnvActor, SubProcessEnv

# Some parameters taken from OpenAI 
# baselines implementation, since 
# they're not mentioned in the paper.
num_actors = 8
# Values taken from Atari experiments
# in original paper where relevant
gae_lambda = 0.95
gamma = 0.99 
base_clip_epsilon = 0.1
max_steps = 1e6
base_learning_rate = 2.5e-4
horizon = 128
batch_size = 32
optim_epochs = 3
value_loss_coefficient = 1
entropy_loss_coefficient = .01
gradient_max = 10.0
start_t = 0
checkpoint_filename = "./ppo-model.ckpt"
log_dir = "./tb_log"
SMALL_NUM = 1e-8

def main():
    env_name = "PongNoFrameskip-v4"
    # NOTE: This is currently not used since we use SubProcessEnv instead;
    # only used for getting shape of observation/acton space.
    unused_env = gym.make(env_name)
    #pobs_shape = unused_env.observation_space.shape
    # Hard-coding pre-processing step shape; could read it from an example output instead?
    obs_shape = (84, 84, 4)
    num_actions = unused_env.action_space.n
    model = SharedModel(obs_shape, num_actions)

    t = start_t
    last_save = 0
    actors = []

    for ii in range(num_actors):
        actors.append(EnvActor(SubProcessEnv(env_name), model, num_actions))

    while(t <= max_steps):
        for ii in range(horizon):
            for actor in actors:
                actor.step_env(t)
            t += 1

        for actor in actors:
            actor.calculate_horizon_advantages(t)

        # Construct randomly sampled (without replacement) mini-batches.
        obs_horizon = []
        act_horizon = []
        policy_horizon = []
        adv_est_horizon = []
        val_est_horizon = []

        for actor in actors:
            obs_a, act_a, policy_a, adv_est_a, val_est_a = actor.get_horizon(t)
            obs_horizon.extend(obs_a)
            act_horizon.extend(act_a)
            policy_horizon.extend(policy_a)
            adv_est_horizon.extend(adv_est_a)
            val_est_horizon.extend(val_est_a)

        # Normalizing advantage estimates.
        # NOTE: Adding this significantly improved performance
        # NOTE: Moved this out of each individual actor, so that advantages for the whole batch are normalized with each other.
        adv_est_horizon = np.array(adv_est_horizon)
        adv_est_horizon = (adv_est_horizon - np.mean(adv_est_horizon)) / (np.std(adv_est_horizon) + SMALL_NUM)

        num_samples = len(obs_horizon)
        indices = list(range(num_samples))
        for e in range(optim_epochs):
            random.shuffle(indices)
            ii = 0
            # TODO: Don't crash if batch_size is not a divisor of total sample count.
            while ii < num_samples:
                obs_batch = []
                act_batch = []
                policy_batch = []
                adv_batch = []
                value_sample_batch = []

                for _ in range(batch_size):
                    index = indices[ii]
                    obs_batch.append(obs_horizon[index].__array__())
                    act_batch.append(act_horizon[index].__array__())
                    policy_batch.append(policy_horizon[index].__array__())
                    adv_batch.append(adv_est_horizon[index].__array__())
                    value_sample_batch.append(val_est_horizon[index])
                    ii += 1                    

                def alpha_anneal(t):
                    return np.maximum(1.0 - (float(t) / float(max_steps)), 0.0)

                total_loss = model.train(np.array(obs_batch),
                            np.array(act_batch), 
                            np.array(policy_batch), 
                            np.array(adv_batch), 
                            np.array(value_sample_batch), 
                            alpha_anneal(t))

        for actor in actors:
            actor.flush(t)

        if t-last_save > 10000:
            print("Saving network")
            model.network.save(checkpoint_filename)
            last_save = t

        all_ep_rewards = []
        for actor in actors:
            all_ep_rewards.extend(actor.episode_rewards)                 

        if len(all_ep_rewards) >= 10:
            print("T: %d" % (t,))
            print("\tAVG Reward: %f" % (np.mean(all_ep_rewards),))
            print("\tMIN Reward: %f" % (np.amin(all_ep_rewards),))
            print("\tMAX Reward: %f" % (np.amax(all_ep_rewards),))
            for actor in actors:
                actor.episode_rewards = []                    

if __name__ == '__main__':
    main()