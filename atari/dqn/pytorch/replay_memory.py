import numpy as np 

class ReplayBuffer():
    """ Simple Replay Buffer implementation for Atari Agents """

    def __init__(self, max_size, input_shape, num_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        # *input_shape allows us to pass in arbitrary sizes for the input shape
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        # this is being done as a way to deal with the terminal reward function being
        # different from the non-terminal reward function.
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
    
    def store_transition(self, state, action, reward, state_, done):
        """ Cicular buffer index iteration and storage of memories """
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[action] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        """ Unique (no repeated samples) batch size sample from memory buffers """
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]
        return states, actions, rewards, states_, dones