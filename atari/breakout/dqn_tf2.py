import gym
import os
import sys
import random
import config
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np 
from gym import wrappers
import matplotlib.pyplot as plt
from datetime import datetime

MAX_EXPERIENCES = 500000
MIN_EXPERIENCES = 50000
TARGET_UPDATE_PERIOD = 10000
IM_SIZE = 84
# Here we are hard-coding the size of the action-space
# as the atari model in gym contains more actions-space
# variables than are useful for this game.
K = 4


class ImageTransformer:
  """
  Transform raw images for input into neural network
  1) Convert to grayscale
  2) Resize
  3) Crop
  """
  def __init__(self):
    init_val = np.ones((210, 160, 3))
    self.input_state = tf.Variable(shape=[210, 160, 3], dtype=tf.uint8, initial_value=init_val)

  @tf.function
  def forward(self):
    self.output = tf.image.rgb_to_grayscale(self.input_state)
    self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
    self.output = tf.image.resize(
        self.output,
        [IM_SIZE, IM_SIZE],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    self.output = tf.squeeze(self.output)
    return self.output 

  def transform(self, state):
    """ Executes transform defined in ImageTransformer """
    self.input_state = state
    return self.forward()

def update_state(state, obs_small):
  """ Takes current state and new frame and returns new state """
  return np.append(state[:,:,1:], np.expand_dims(obs_small, 2), axis=2)

class ReplayMemory:
  def __init__(self, size=MAX_EXPERIENCES, frame_height=IM_SIZE, frame_width=IM_SIZE, 
               agent_history_length=4, batch_size=32):
    """
    Args:
        size: Integer, Number of stored transitions
        frame_height: Integer, Height of a frame of an Atari game
        frame_width: Integer, Width of a frame of an Atari game
        agent_history_length: Integer, Number of frames stacked together to create a state
        batch_size: Integer, Number of transitions returned in a minibatch
    """
    self.size = size
    self.frame_height = frame_height
    self.frame_width = frame_width
    self.agent_history_length = agent_history_length
    self.batch_size = batch_size
    self.count = 0
    self.current = 0

    # Pre-allocate memory
    self.actions = np.empty(self.size, dtype=np.int32)
    self.rewards = np.empty(self.size, dtype=np.float32)
    self.frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.uint8)
    self.terminal_flags = np.empty(self.size, dtype=np.bool)

    # Pre-allocate memory for the states and new_states in a minibatch
    self.states = np.empty((self.batch_size, self.agent_history_length, 
                            self.frame_height, self.frame_width), dtype=np.uint8)
    self.new_states = np.empty((self.batch_size, self.agent_history_length, 
                                self.frame_height, self.frame_width), dtype=np.uint8)
    self.indices = np.empty(self.batch_size, dtype=np.int32)

  def add_experience(self, action, frame, reward, terminal):
    """
    Circular buffer experience adder, self.current marks
    current index circular buffer is writing into.
    Args:
        action: An integer-encoded action
        frame: One grayscale frame of the game
        reward: reward the agend received for performing an action
        terminal: A bool stating whether the episode terminated
    """
    if frame.shape != (self.frame_height, self.frame_width):
      raise ValueError('Dimension of frame is wrong!')
    self.actions[self.current] = action
    self.frames[self.current, ...] = frame
    self.rewards[self.current] = reward
    self.terminal_flags[self.current] = terminal
    self.count = max(self.count, self.current+1)
    self.current = (self.current + 1) % self.size

  def _get_state(self, index):
    """ Returns state given end index of state """
    if self.count is 0:
      raise ValueError("The replay memory is empty!")
    if index < self.agent_history_length - 1:
      raise ValueError("Index must be min {}".format(self.agent_history_length-1))
    return self.frames[index-self.agent_history_length+1:index+1, ...]

  def _get_valid_indices(self):
    """ Helper function for sampling a batch, assigns indicies
    to an instance variable """
    for i in range(self.batch_size):
      while True:
        # Sample a random index
        index = random.randint(self.agent_history_length, self.count - 1)
        # Ensure that the index we have sampled is valid
        # 1 ) Must have valid states
        # 2 ) Must not sample within 3 frames ahead of self.current
        # 3 ) Must not be at the boundary of an episode (done flag)
        if index < self.agent_history_length:
          continue
        if index >= self.current and index - self.agent_history_length <= self.current:
          continue
        if self.terminal_flags[index - self.agent_history_length:index].any():
          continue
        break
      self.indices[i] = index

  def get_minibatch(self):
    """
    Returns a minibatch of self.batch_size transitions
    """
    if self.count < self.agent_history_length:
      raise ValueError('Not enough memories to get a minibatch')
    
    self._get_valid_indices()
        
    # Get states for valid indicies
    for i, idx in enumerate(self.indices):
      self.states[i] = self._get_state(idx - 1)
      self.new_states[i] = self._get_state(idx)
    
    # Return the minibatch, (S, A, S', R)
    return np.transpose(self.states, axes=(0, 2, 3, 1)), self.actions[self.indices], self.rewards[self.indices], np.transpose(self.new_states, axes=(0, 2, 3, 1)), self.terminal_flags[self.indices]

class DQN:
  def __init__(self, K, conv_layer_sizes, hidden_layer_sizes):

    self.K = K
    self.X = keras.Input(dtype=tf.float32, shape=(IM_SIZE, IM_SIZE, 4), name='X')
    self.z = tf.keras.layers.Conv2D(32, 8, strides=4, activation='relu')(self.X)
    self.z = tf.keras.layers.Conv2D(64, 4, strides=2, activation='relu')(self.z)
    self.z = tf.keras.layers.Conv2D(64, 3, strides=1, activation='relu')(self.z)
    self.z = tf.keras.layers.Flatten()(self.z)
    self.z = tf.keras.layers.Dense(512)(self.z)
    self.z = tf.keras.layers.Dense(self.K)(self.z)
    self.Y = self.z

    self.network = tf.keras.Model(inputs=[self.X], outputs=[self.Y])
    self.weights = self.network.trainable_variables
    self.optimizer = tf.keras.optimizers.Adam(1e-5)

  def copy_from(self, other):
    """ Copy weights from one network to another network, 
    we need this to copy weights to our target network """
    self.network.set_weights(other.network.get_weights())

  def save(self):
    self.network.save_weights('tf_dqn.h5')

  def load(self):
    self.network.load_weights('tf_dqn.h5')

  def predict(self, states):
    return self.network(tf.cast(states, tf.float32))

  def update(self, states, actions, targets):
    with tf.GradientTape() as t:
      selected_action_values = tf.reduce_sum(self.predict(states)*keras.backend.one_hot(actions, K), axis=1)
      cost = tf.reduce_mean(tf.compat.v1.losses.huber_loss(targets, selected_action_values))
    grads = t.gradient(cost, self.weights)
    self.optimizer.apply_gradients(zip(grads, self.weights))
    return cost

  def sample_action(self, x, eps):
    # e-greedy
    if np.random.random() < eps:
      return np.random.choice(self.K)
    else:
      return np.argmax(self.predict([x])[0])

def learn(model, target_model, experience_replay_buffer, gamma, batch_size):
  # Sample experiences -> S, A, R, S'
  states, actions, rewards, next_states, dones = experience_replay_buffer.get_minibatch()
  # Calculate targets
  next_Qs = target_model.predict(next_states)
  # Act greedily to select A'
  next_Q = np.amax(next_Qs, axis=1)
  # G = R if Done is True, Else, G = R + Gamma*Q'
  targets = rewards + np.invert(dones).astype(np.float32) * gamma * next_Q
  # Update model, return loss
  loss = model.update(states, actions, targets)
  return loss

def play_one(
  env,
  total_t,
  experience_replay_buffer,
  model,
  target_model,
  image_transformer,
  gamma,
  batch_size,
  epsilon,
  epsilon_change,
  epsilon_min):
  """ 
  Plays one episode of the game.
  Args:
  Total T: Total steps played so far.
  """

  t0 = datetime.now()

  # Reset the environment
  obs = env.reset()
  obs_small = image_transformer.transform(obs)
  state = np.stack([obs_small] * 4, axis=2)

  total_time_training = 0
  num_steps_in_episode = 0
  episode_reward = 0

  done = False
  while not done:

    # Update target network on update period
    if total_t % TARGET_UPDATE_PERIOD == 0:
      target_model.copy_from(model)
      print("Copied model parameters to target network. total_t = %s, period = %s" % (total_t, TARGET_UPDATE_PERIOD))

    # Take action
    action = model.sample_action(state, epsilon)
    obs, reward, done, _ = env.step(action)
    # Downsample observation to appropriate size for our network
    obs_small = image_transformer.transform(obs)
    # State requires multiple frames, so we need to construct this here
    next_state = update_state(state, obs_small)

    # Compute total reward
    episode_reward += reward

    # Save the latest experience to the replay buffer
    experience_replay_buffer.add_experience(action, obs_small, reward, done)    

    # Train the model, keep track of time
    t0_2 = datetime.now()
    loss = learn(model, target_model, experience_replay_buffer, gamma, batch_size)
    dt = datetime.now() - t0_2

    # More debugging info
    total_time_training += dt.total_seconds()
    num_steps_in_episode += 1

    state = next_state
    total_t += 1

    epsilon = max(epsilon - epsilon_change, epsilon_min)

  return total_t, episode_reward, (datetime.now() - t0), num_steps_in_episode, total_time_training/num_steps_in_episode, epsilon

def smooth(x):
  """ Helper function to keep track of the average return
  over the last 100 episodes """
  # last 100
  n = len(x)
  y = np.zeros(n)
  for i in range(n):
    start = max(0, i - 99)
    y[i] = float(x[start:(i+1)].sum()) / (i - start + 1)
  return y

if __name__ == '__main__':

  # hyperparams and initialize stuff
  conv_layer_sizes = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
  hidden_layer_sizes = [512]
  gamma = 0.99
  batch_sz = 32
  num_episodes = 3500
  total_t = 0
  experience_replay_buffer = ReplayMemory()
  episode_rewards = np.zeros(num_episodes)

  # epsilon
  # decays linearly until 0.1
  epsilon = 1.0
  epsilon_min = 0.1
  epsilon_change = (epsilon - epsilon_min) / 500000

  # Create environment
  env = gym.envs.make("Breakout-v0")

  # Create models
  model = DQN(
    K=K,
    conv_layer_sizes=conv_layer_sizes,
    hidden_layer_sizes=hidden_layer_sizes
    )
  target_model = DQN(
    K=K,
    conv_layer_sizes=conv_layer_sizes,
    hidden_layer_sizes=hidden_layer_sizes
    )
  image_transformer = ImageTransformer()

  print("Populating experience replay buffer...")
  obs = env.reset()

  # Fill experience replay buffer with random actions
  for i in range(MIN_EXPERIENCES):

      action = np.random.choice(K)
      obs, reward, done, _ = env.step(action)
      obs_small = image_transformer.transform(obs)
      experience_replay_buffer.add_experience(action, obs_small, reward, done)
      # reset if env ends
      if done:
          obs = env.reset()

  # Play a number of episodes and learn!
  t0 = datetime.now()
  for i in range(num_episodes):

    total_t, episode_reward, duration, num_steps_in_episode, time_per_step, epsilon = play_one(
      env,
      total_t,
      experience_replay_buffer,
      model,
      target_model,
      image_transformer,
      gamma,
      batch_sz,
      epsilon,
      epsilon_change,
      epsilon_min,
    )
    episode_rewards[i] = episode_reward

    last_100_avg = episode_rewards[max(0, i - 100):i + 1].mean()
    print("Episode:", i,
      "Duration:", duration,
      "Num steps:", num_steps_in_episode,
      "Reward:", episode_reward,
      "Training time per step:", "%.3f" % time_per_step,
      "Avg Reward (Last 100):", "%.3f" % last_100_avg,
      "Epsilon:", "%.3f" % epsilon
    )
    sys.stdout.flush()
  print("Total duration:", datetime.now() - t0)

  model.save()

  # Plot the smoothed returns
  y = smooth(episode_rewards)
  plt.plot(episode_rewards, label='orig')
  plt.plot(y, label='smoothed')
  plt.legend()
  plt.show()