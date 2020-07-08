import numpy as np
from wrappers import make_atari, wrap_deepmind

start_t = 0
gamma = 0.99 

def indicator(x):
    if x:
        return 1
    else:
        return 0

# Data relevant to executing and collecting samples from a single environment execution.
class EnvActor(object):
    def __init__(self, env, network):
        self.env = env
        self.network = network
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

    def step_env(self, t):
        if t == start_t:
            # Artifact of ordering
            #val_0 = sess.run(value, {obs_ph: [self.last_obs]})
            #self.val.append(val_0[0])
            value_0, _ = self.network(self.last_obs)
            self.val.append(value_0[0])
        
        _, policy_t = self.network(self.last_obs)
        #policy_t = sess.run(policy, {obs_ph: [self.last_obs]})[0]
        
        action_t = np.random.choice(self.env.action_space.n, 1, p=policy)[0]
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
        # val_tp1 = sess.run(value, {obs_ph: [obs_tp1]})
        val_tp1, _ = self.network(obs_tp1)
        self.val.append(val_tp1[0])  
        self.delta.append(self.rew.get(t) +  (1 - indicator(done_t)) * gamma * self.val.get(t + 1) - self.val.get(t))

        self.last_obs = obs_tp1

# Creates a gym environment in a sub-process.  Necessary because Gym does not allow multiple concurrent environments in the same process.
class SubProcessEnv(object):
    def env_process(name, conn):
        env = make_atari(name)
        env = wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=True, scale=True)
        while True:
            (cmd, args, kwargs) = conn.recv()
            if cmd == "reset":
                conn.send(env.reset())
            elif cmd == "step":
                conn.send(env.step(*args, **kwargs))
            elif cmd == "exit":
                break
            else:
                raise Exception("Unknown command %s" % (str(cmd),))

    def __init__(self, name):
        parent_conn, child_conn = Pipe()
        self.process = Process(target=SubProcessEnv.env_process, args=(name, child_conn))
        self.process.start()
        self.conn = parent_conn

    def step(self, action):
        self.conn.send(("step", (action,), {}))
        return self.conn.recv()

    def reset(self):
        self.conn.send(("reset", (), {}))
        return self.conn.recv()

    def exit(self):
        self.conn.send(("exit", (), {}))
        self.process.join()

class TimeIndexedList(object):
    def __init__(self, first_t=0):
        self.first_t = first_t
        self.list = []

    # For flushing so that we don't keep unnecessary history forever.
    def flush_through(self, t):
        to_remove = t - self.first_t + 1
        if to_remove > 0:
            self.list = self.list[to_remove:]
            self.first_t = t + 1

    def append(self, elem):
        self.list.append(elem)

    def get(self, t):
        return self.list[t - self.first_t]

    def future_length(self):
        return len(self.list)

    def get_range(self, t, length):
        return self.list[(t - self.first_t):(t - self.first_t + length)]

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
