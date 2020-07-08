from wrappers import make_atari, wrap_deepmind
import gym
#import matploblib.pyplot as plt
from multiprocessing import Process, Pipe
import numpy as np
import random
from skimage.viewer import ImageViewer
from skimage.transform import resize
import tensorflow as tf
import tensorflow.contrib.layers as layers

# Some parameters taken from OpenAI baselines implementation, since they're not mentioned in the paper.
num_actors = 8
# Values taken from Atari experiments in original paper where relevant
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

env_name = "PongNoFrameskip-v4"
# NOTE: This is currently not used since we use SubProcessEnv instead; only used for getting shape of observation/acton space.
unused_env = gym.make(env_name)
#pobs_shape = unused_env.observation_space.shape
# Hard-coding pre-processing step shape; could read it from an example output instead?
obs_shape = (84, 84, 4)
num_actions = unused_env.action_space.n
discrete = True




# Data relevant to executing and collecting samples from a single environment execution.
class EnvActor(object):
    def __init__(self, env):
        self.env = env
        self.obs = TimeIndexedList(first_t = start_t)
        self.last_obs = self.env.reset()
        self.obs.append(self.last_obs)
        # self.pobs = TimeIndexedList(first_t = start_t)
        # self.last_pobs = preprocess_obs_atari(self.obs, self.pobs, start_t, start_t)
        # self.pobs.append(self.last_pobs)
        self.act = TimeIndexedList(first_t = start_t)
        self.rew = TimeIndexedList(first_t = start_t)
        self.val = TimeIndexedList(first_t = start_t)
        self.policy = TimeIndexedList(first_t = start_t)
        self.delta = TimeIndexedList(first_t = start_t)
        self.done = TimeIndexedList(first_t = start_t)
        self.episode_start_t = 0
        self.episode_rewards = []
        self.rewards_this_episode = []
        self.advantage_estimates = TimeIndexedList(first_t = start_t)
        self.value_estimates = TimeIndexedList(first_t = start_t)

    def step_env(self, sess, logger, obs_ph, policy, value, t):
        if t == start_t:
            # Artifact of ordering
            val_0 = sess.run(value, {obs_ph: [self.last_obs]})
            self.val.append(val_0[0])

        policy_t = sess.run(policy, {obs_ph: [self.last_obs]})[0]
        # NOTE: Huge bug -- was deterministically choosing the action, instead of stochastically!
        # action_t = np.argmax(policy_t):
        action_t = np.random.choice(num_actions, 1, p=policy_t)[0]
        obs_tp1, rew_t, done_t, unused_info = self.env.step(action_t)
        self.act.append(action_t)
        self.rew.append(rew_t)
        self.policy.append(policy_t)
        self.rewards_this_episode.append(rew_t)

        if done_t:
            ep_sum = tf.Summary()
            ep_sum.value.add(tag="episode_reward", simple_value=sum(self.rewards_this_episode))
            ep_sum.value.add(tag="episode_length", simple_value=len(self.rewards_this_episode))
            logger.add_summary(ep_sum, t)

            self.done.append(True)
            self.episode_rewards.append(sum(self.rewards_this_episode))
            self.rewards_this_episode = []
            obs_tp1 = self.env.reset()
            self.episode_start_t = t + 1
        else:
            self.done.append(False)

        # Important to put this after we've updated obs_tp1 in case of reset.
        # NOTE: Bug fix, obs_horizon was being added to before the possible reset, so wrong observation was associated to initial policy step.
        self.obs.append(obs_tp1)
        # pobs_tp1 = preprocess_obs_atari(self.obs, self.pobs, t + 1, self.episode_start_t)
        # self.pobs.append(pobs_tp1)
        val_tp1 = sess.run(value, {obs_ph: [obs_tp1]})
        self.val.append(val_tp1[0])  
        self.delta.append(self.rew.get(t) +  (1 - indicator(done_t)) * gamma * self.val.get(t + 1) - self.val.get(t))

        self.last_obs = obs_tp1

    # end_t is non-inclusive, i.e. it's the t immediately after the desired horizon.
    def calculate_horizon_advantages(self, end_t):
        advantage_estimates = []
        value_estimates = []
        advantage_so_far = 0
        # No empirical estimate beyond end of horizon, so use value function.  Is immediately reset to 0 if at episode boundary.
        last_value_sample = self.val.get(end_t)
        for ii in range(horizon):
            if self.done.get(end_t - ii - 1):
                advantage_so_far = 0
                last_value_sample = 0
            
            # Insert in reverse order.
            advantage_so_far = self.delta.get(end_t - ii - 1) + (gamma * gae_lambda * advantage_so_far)
            advantage_estimates.append(advantage_so_far)
            # NOTE: Was using 1-step value update here; instead use the GAE value estimate (i.e. Q(s,a) with the empirical action.)
            #last_value_sample = (1 - indicator(self.done.get(end_t - ii - 1))) * gamma * last_value_sample + self.rew.get(end_t - ii - 1)
            # Didn't need the 1 - indicator since setting this above.
            last_value_sample = gamma * last_value_sample + self.rew.get(end_t - ii - 1)
            value_estimates.append(last_value_sample)
            #value_estimates.append(advantage_so_far + self.val.get(end_t - ii - 1))
            
            #value_sample_estimates.append((1 - indicator(done_horizon.get(t - ii - 1))) * gamma * val_horizon.get(t - ii)  + rew_horizon.get(t - ii - 1)) 
        advantage_estimates.reverse()
        value_estimates.reverse()

        # NOTE: Was normalizing here, but moved that to whole batch.
        for ii in range(len(advantage_estimates)):
            self.advantage_estimates.append(advantage_estimates[ii])
            self.value_estimates.append(value_estimates[ii])


    def get_horizon(self, end_t):
        return (self.obs.get_range(end_t - horizon, horizon),
                self.act.get_range(end_t - horizon, horizon),
                self.policy.get_range(end_t - horizon, horizon),
                self.advantage_estimates.get_range(end_t - horizon, horizon),
                self.value_estimates.get_range(end_t - horizon, horizon))

    def flush(self, end_t):
        # Retain some extra observations for preprocessing step.
        self.obs.flush_through(end_t - horizon - 5)
        self.act.flush_through(end_t - horizon - 1)
        self.rew.flush_through(end_t - horizon - 1)
        self.val.flush_through(end_t - horizon - 1)
        self.policy.flush_through(end_t - horizon - 1)
        self.delta.flush_through(end_t - horizon - 1)
        self.done.flush_through(end_t - horizon - 1)



def shared_network_cartpole(input):
    out = layers.fully_connected(input, num_outputs=32, activation_fn=tf.nn.relu)
    out = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.relu)
    return out

# The architecture used for the PPO paper isn't described there; had to trace back to "Human-level control through deep reinforcement learning" Mnih et al 2015
# Is also the architecture used in CS294 Au 2017 HW3
# May or may not also use a layered 256 unit LSTM at the end (mentioned in intermediate referenced paper).
def shared_network_atari(input):
    out = layers.convolution2d(input, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
    out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
    out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
    out = layers.flatten(out)
    return layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)

def value_tail(shared):
    return tf.squeeze(layers.fully_connected(shared, num_outputs=1, activation_fn=None), axis=1)

def policy_tail(shared):
    return tf.nn.softmax(layers.fully_connected(shared, num_outputs=num_actions, activation_fn=None), axis=1)

def alpha_anneal(t):
    return np.maximum(1.0 - (float(t) / float(max_steps)), 0.0)

def indicator(x):
    if x:
        return 1
    else:
        return 0

def train():
    sess = tf.Session()
    with sess:
        obs_ph = tf.placeholder(tf.float32, shape=(None,) + obs_shape)
        alpha_ph = tf.placeholder(tf.float32, shape=())

        shared_net = shared_network_atari(obs_ph)
        value_net = value_tail(shared_net)
        policy = policy_tail(shared_net)

        act_ph = tf.placeholder(tf.int32, shape=(None,))
        act_onehot = tf.one_hot(act_ph, num_actions)
        policy_old_ph = tf.placeholder(tf.float32, shape=(None, num_actions))

        clip_epsilon = alpha_ph * base_clip_epsilon
        # NOTE: Forgot to add axis=1 here, huge bug.
        log_prob_ratio = tf.log(tf.reduce_sum(policy * act_onehot, axis=1)) - tf.log(tf.reduce_sum(policy_old_ph * act_onehot, axis=1))
        prob_ratio = tf.exp(log_prob_ratio)
        #prob_ratio = tf.reduce_sum(policy * act_onehot, axis=1) / (tf.reduce_sum(policy_old_ph * act_onehot, axis=1) + SMALL_NUM)
        clipped_prob_ratio = tf.clip_by_value(prob_ratio, 1 - clip_epsilon, 1 + clip_epsilon)
        advantage_estimate_ph = tf.placeholder(tf.float32, shape=(None,))
        value_sample_estimate_ph = tf.placeholder(tf.float32, shape=(None,))

        # NOTE: Needed to be negative, since we want entropy to be big, not small.
        entropy_loss = -tf.reduce_sum(-policy * tf.log(policy), axis=1)
        # NOTE: Huge bug, needed to maximize the minimum, not minimize it.  Without the minus sign, is not a loss.
        clip_loss = -tf.minimum(prob_ratio * advantage_estimate_ph, clipped_prob_ratio * advantage_estimate_ph)
        value_loss = tf.square(value_net - value_sample_estimate_ph)
        total_loss = tf.reduce_mean(clip_loss + value_loss_coefficient * value_loss + entropy_loss_coefficient * entropy_loss)

        learning_rate = base_learning_rate * alpha_ph
        #update_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
        # NOTE: Took epsilon from OpenAI Baselines ppo2.
        adam = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-5)
        gradients, variables = zip(*adam.compute_gradients(total_loss))
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, gradient_max)
        update_op = adam.apply_gradients(zip(clipped_gradients, variables))

        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        # Uncomment to resume from last checkpoint.
        # saver.restore(sess, checkpoint_filename)
        logger = tf.summary.FileWriter(log_dir, sess.graph)
        tf.summary.histogram("total_loss", total_loss)
        tf.summary.histogram("value_loss", value_loss)
        tf.summary.histogram("clip_loss", clip_loss)
        tf.summary.histogram("entropy_loss", entropy_loss)
        summary_op = tf.summary.merge_all()

        # Global time counter
        # At time t, take a_t from s_t, receive r_t.
        t = start_t
        last_save = 0

        actors = []
        for ii in range(num_actors):
            actors.append(EnvActor(SubProcessEnv(env_name)))

        while t <= max_steps:
            for ii in range(horizon):
                for actor in actors:
                    actor.step_env(sess, logger, obs_ph, policy, value_net, t)
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
            # NOTE:  Adding this significantly improved performance
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

                    for b in range(batch_size):
                        index = indices[ii]
                        obs_batch.append(obs_horizon[index])
                        act_batch.append(act_horizon[index])
                        policy_batch.append(policy_horizon[index])
                        adv_batch.append(adv_est_horizon[index])
                        value_sample_batch.append(val_est_horizon[index])
                        ii += 1                    

                    update_feed = {
                        obs_ph: obs_batch,
                        act_ph: act_batch,
                        policy_old_ph: policy_batch,
                        advantage_estimate_ph: adv_batch,
                        value_sample_estimate_ph: value_sample_batch,
                        alpha_ph: alpha_anneal(t)
                    }
                    _, summary = sess.run([update_op, summary_op], update_feed)
                    logger.add_summary(summary, t)


            for actor in actors:
                actor.flush(t)

            if t - last_save > 10000:
                saver.save(sess, checkpoint_filename)
                last_save = t

            all_ep_rewards = []
            for actor in actors:
                all_ep_rewards.extend(actor.episode_rewards)

            if len(all_ep_rewards) >= 10:
                print("T: %d" % (t,))
                print("AVG Reward: %f" % (np.mean(all_ep_rewards),))
                print("MIN Reward: %f" % (np.amin(all_ep_rewards),))
                print("MAX Reward: %f" % (np.amax(all_ep_rewards),))
                for actor in actors:
                    actor.episode_rewards = []
                # print("Entropy Loss: %f" % (np.mean(eloss),))
                # print("Value Loss: %f" % (np.mean(vloss),))
                # print("Clip Loss: %f" % (np.mean(closs),))
                # print("Total Loss: %f" % (np.mean(tloss),))

train()