# https://deeplearningcourses.com/c/cutting-edge-artificial-intelligence
import numpy as np
import tensorflow as tf
from tensorflow import keras
import gym
import matplotlib.pyplot as plt
from datetime import datetime


### avoid crashing on Mac
# doesn't seem to work
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")


class ANN:
  
  def __init__(self, x, output_size, hidden_sizes, hidden_activation, output_activation, lr):
    for h in hidden_sizes:
      x = keras.layers.Dense(units=h, activation=hidden_activation)(x)
    self.output = keras.layers.Dense(units=output_size, activation=output_activation)(x)
  
class Mu(ANN):

  def __init__(self, x, output_size, hidden_sizes, hidden_activation, output_activation, lr, action_max):
    self.input = x
    ANN.__init__(self, x, output_size, hidden_sizes, hidden_activation, output_activation, lr)
    self.action_max = action_max
    self.network = tf.keras.Model(inputs=self.input, outputs=self.output)
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

  def forward(self, state):
    return self.action_max * self.network(state)

  def train(self, state, q_net):
    with tf.GradientTape() as t:
        mu_s = self.forward(state)
        q_mu = q_net.forward([state, mu_s])
        mu_loss = -tf.reduce_mean(q_mu)
    grads = t.gradient(mu_loss, self.network.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.network.trainable_weights))
    return mu_loss


class Q(ANN):

  def __init__(self, inputs, output_size, hidden_sizes, hidden_activation, output_activation, lr):
    self.input = inputs
    x = keras.layers.concatenate(inputs, axis=-1)
    ANN.__init__(self, x, output_size, hidden_sizes, hidden_activation, output_activation, lr)
    self.network = tf.keras.Model(inputs=self.input, outputs=self.output)
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

  def forward(self, inputs):
    return tf.squeeze(self.network(inputs), axis=1)

  def train(self, inputs, q_targets):
    with tf.GradientTape() as t:
      q_loss = tf.reduce_mean((self.forward(inputs) - q_targets)**2)
    grads = t.gradient(q_loss, self.network.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.network.trainable_weights))
    return q_loss


### The experience replay memory ###
class ReplayBuffer:
  def __init__(self, obs_dim, act_dim, size):
    self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
    self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
    self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
    self.rews_buf = np.zeros(size, dtype=np.float32)
    self.done_buf = np.zeros(size, dtype=np.float32)
    self.ptr, self.size, self.max_size = 0, 0, size

  def store(self, obs, act, rew, next_obs, done):
    self.obs1_buf[self.ptr] = obs
    self.obs2_buf[self.ptr] = next_obs
    self.acts_buf[self.ptr] = act
    self.rews_buf[self.ptr] = rew
    self.done_buf[self.ptr] = done
    self.ptr = (self.ptr+1) % self.max_size
    self.size = min(self.size+1, self.max_size)

  def sample_batch(self, batch_size=32):
    idxs = np.random.randint(0, self.size, size=batch_size)
    return dict(s=self.obs1_buf[idxs],
                s2=self.obs2_buf[idxs],
                a=self.acts_buf[idxs],
                r=self.rews_buf[idxs],
                d=self.done_buf[idxs])


### Implement the DDPG algorithm ###
def ddpg(
    env_fn,
    hidden_sizes,
    seed=0,
    save_folder=None,
    num_train_episodes=100,
    test_agent_every=25,
    replay_size=int(1e6),
    gamma=0.99, 
    decay=0.95,
    mu_lr=1e-4,
    q_lr=1e-4,
    batch_size=100,
    start_steps=10000, 
    action_noise=0.1,
    max_episode_length=1000):

  tf.random.set_seed(seed)
  np.random.seed(seed)

  env, test_env = env_fn(), env_fn()

  # comment out this line if you don't want to record a video of the agent
  if save_folder is not None:
    test_env = gym.wrappers.Monitor(test_env, save_folder, force=True)

  # get size of state space and action space
  num_states = env.observation_space.shape[0]
  num_actions = env.action_space.shape[0]

  # Maximum value of action
  # Assumes both low and high values are the same
  # Assumes all actions have the same bounds
  # May NOT be the case for all environments
  action_max = env.action_space.high[0]

  # Create Tensorflow placeholders (neural network inputs)
  X = keras.Input(shape=[num_states], dtype=tf.float32) # state
  A = keras.Input(shape=[num_actions], dtype=tf.float32) # action

  # X2 = tf.placeholder(dtype=tf.float32, shape=(None, num_states)) # next state
  # R = tf.placeholder(dtype=tf.float32, shape=(None,)) # reward
  # D = tf.placeholder(dtype=tf.float32, shape=(None,)) # done

  # Main network outputs
  mu_net = Mu(X, num_actions, hidden_sizes, keras.activations.relu, keras.activations.tanh, mu_lr, action_max)
  q_net = Q([X, A], 1, hidden_sizes, keras.activations.relu, None, q_lr)
 
  # We don't need the Q network output with arbitrary input action A
  # because that's not actually used in our loss functions
  # NOTE 1: The state input is X2, NOT X
  #         We only care about max_a{ Q(s', a) }
  #         Where this is equal to Q(s', mu(s'))
  #         This is because it's used in the target calculation: r + gamma * max_a{ Q(s',a) }
  #         Where s' = X2
  # NOTE 2: We ignore the first 2 networks for the same reason
  mu_target_net = Mu(X, num_actions, hidden_sizes, keras.activations.relu, keras.activations.tanh, mu_lr, action_max)
  q_target_net = Q([X, A], 1, hidden_sizes, keras.activations.relu, None, q_lr)

  # Experience replay memory
  replay_buffer = ReplayBuffer(obs_dim=num_states, act_dim=num_actions, size=replay_size)
  
  mu_target_net.network.set_weights(mu_net.network.get_weights())
  q_target_net.network.set_weights(q_net.network.get_weights())

  def get_action(s, noise_scale):
    a = mu_net.forward(s.reshape(1,-1))[0]
    #a = sess.run(mu, feed_dict={X: s.reshape(1,-1)})[0]
    a += noise_scale * np.random.randn(num_actions)
    return np.clip(a, -action_max, action_max)

  test_returns = []
  def test_agent(num_episodes=5):
    t0 = datetime.now()
    n_steps = 0
    for _ in range(num_episodes):
      s, episode_return, episode_length, d = test_env.reset(), 0, 0, False
      while not (d or (episode_length == max_episode_length)):
        # Take deterministic actions at test time (noise_scale=0)
        test_env.render()
        s, r, d, _ = test_env.step(get_action(s, 0))
        episode_return += r
        episode_length += 1
        n_steps += 1
      print('\tTest Return:', episode_return, '\tEpisode Length:', episode_length)
      test_returns.append(episode_return)
    # print("test steps per sec:", n_steps / (datetime.now() - t0).total_seconds())


  # Main loop: play episode and train
  returns = []
  q_losses = []
  mu_losses = []
  num_steps = 0
  for i_episode in range(num_train_episodes):

    # reset env
    s, episode_return, episode_length, d = env.reset(), 0, 0, False

    while not (d or (episode_length == max_episode_length)):
      # For the first `start_steps` steps, use randomly sampled actions
      # in order to encourage exploration.
      if num_steps > start_steps:
        a = get_action(s, action_noise)
      else:
        a = env.action_space.sample()

      # Keep track of the number of steps done
      num_steps += 1
      if num_steps == start_steps:
        print("USING AGENT ACTIONS NOW")

      # Step the env
      s2, r, d, _ = env.step(a)
      episode_return += r
      episode_length += 1

      # Ignore the "done" signal if it comes from hitting the time
      # horizon (that is, when it's an artificial terminal signal
      # that isn't based on the agent's state)
      d_store = False if episode_length == max_episode_length else d

      # Store experience to replay buffer
      replay_buffer.store(s, a, r, s2, d_store)

      # Assign next state to be the current state on the next round
      s = s2

    # Perform the updates
    for _ in range(episode_length):
      batch = replay_buffer.sample_batch(batch_size)

      q_target = batch['r'] * gamma * (1-batch['d'])
      q_target *= q_target_net.forward([batch['s2'], batch['a']])

      # Q network update
      ql = q_net.train([batch['s'], batch['a']], q_target)
      q_losses.append(ql)

      # Policy update
      # (And target networks update)
      # Note: plot the mu loss if you want
      mul =  mu_net.train(batch['s'], q_net)
      mu_losses.append(mul)
      # Use soft updates to update the target networks

      mu_decay_target = [decay*x for x in mu_target_net.network.get_weights()]
      mu_decay_update = [(1-decay)*x for x in mu_net.network.get_weights()]
      mu_updated_weights = [sum(x) for x in zip(mu_decay_target, mu_decay_update)]
      
      # print("Orig")
      # print(mu_target_net.network.get_weights()[0][:,0])
      # print("Updater")
      # print(mu_net.network.get_weights()[0][:,0])
      # print("New")
      # print(mu_updated_weights[0][:,0])
      # print("\n\n")
      # quit()

      mu_target_net.network.set_weights(mu_updated_weights)

      q_decay_target = [decay*x for x in q_target_net.network.get_weights()]
      q_decay_update = [(1-decay)*x for x in q_net.network.get_weights()]
      q_updated_weights = [sum(x) for x in zip(q_decay_target, q_decay_update)]
      
      q_target_net.network.set_weights(q_updated_weights)

    print("Episode:", i_episode + 1, "\tReturn:", episode_return, '\tEpisode Length:', episode_length)
    returns.append(episode_return)

    # Test the agent
    if i_episode > 0 and i_episode % test_agent_every == 0:
      test_agent()

  # on Mac, plotting results in an error, so just save the results for later
  # if you're not on Mac, feel free to uncomment the below lines
  np.savez('ddpg_results.npz', train=returns, test=test_returns, q_losses=q_losses, mu_losses=mu_losses)

  # plt.plot(returns)
  # plt.plot(smooth(np.array(returns)))
  # plt.title("Train returns")
  # plt.show()

  # plt.plot(test_returns)
  # plt.plot(smooth(np.array(test_returns)))
  # plt.title("Test returns")
  # plt.show()

  # plt.plot(q_losses)
  # plt.title('q_losses')
  # plt.show()

  # plt.plot(mu_losses)
  # plt.title('mu_losses')
  # plt.show()


def smooth(x):
  # last 100
  n = len(x)
  y = np.zeros(n)
  for i in range(n):
    start = max(0, i - 99)
    y[i] = float(x[start:(i+1)].sum()) / (i - start + 1)
  return y


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  # parser.add_argument('--env', type=str, default='HalfCheetah-v2')
  parser.add_argument('--env', type=str, default='Pendulum-v0')
  parser.add_argument('--hidden_layer_sizes', type=int, default=300)
  parser.add_argument('--num_layers', type=int, default=1)
  parser.add_argument('--gamma', type=float, default=0.99)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--num_train_episodes', type=int, default=200)
  parser.add_argument('--save_folder', type=str, default='ddpg_monitor')
  args = parser.parse_args()


  ddpg(
    lambda : gym.make(args.env),
    hidden_sizes=[args.hidden_layer_sizes]*args.num_layers,
    gamma=args.gamma,
    seed=args.seed,
    save_folder=args.save_folder,
    num_train_episodes=args.num_train_episodes,
  )
