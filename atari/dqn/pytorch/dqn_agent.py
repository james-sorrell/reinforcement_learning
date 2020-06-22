import numpy as np
import torch as T
from dqn_pytorch import DeepQNetwork
from replay_memory import ReplayBuffer

class DQNAgent():
    def __init__(self, gamma, epsilon, lr, num_actions, input_dims, mem_size, 
        batch_size, eps_min=0.01, eps_start=5e-7, replace=1000, algo=None, 
        env_name=None, checkpoint_dir='data'):
        """ DQN Agent Parameter, Replay Buffer and Env Declaration """
                
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.num_actions = num_actions
        self.input_dimensions = input_dims
        self.batch_size = batch_size
        self.epsilon_min = eps_min
        self.epsilon_max = eps_start
        self.replace_target_count = replace
        self.algo = algo
        self.env_name = env_name
        self.checkpoint_dir = checkpoint_dir
        self.action_space = [i for i in range(self.num_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, self.input_dimensions, self.num_actions)

        self.q_eval = DeepQNetwork(name=self.env_name+'_'+self.algo+'_q_eval',
                            input_dims=self.input_dimensions,
                            checkpoint_dir=self.checkpoint_dir,
                            num_actions=self.num_actions,
                            lr=self.lr)

        self.q_targ = DeepQNetwork(name=self.env_name+'_'+self.algo+'_q_targ',
                            input_dims=self.input_dimensions,
                            checkpoint_dir=self.checkpoint_dir,
                            num_actions=self.num_actions,
                            lr=self.lr)

    def choose_action(self, observation):
        """ Epsilon greedy action selection """
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def store_transition(self, state, action, reward, state_, done):
        """ Store memory sample into replay buffer """
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        """ Sample replay buffer and convert the batch into buffers for inference """
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        states = T.tensor(state).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        return states, actions, rewards, states_, dones
        
    def replace_target_network(self):
        """ Copy weights from trained network to target network """
        if self.learn_step_counter % self.replace_target_count == 0:
            self.q_targ.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon - self.epsilon_max 
        else:
            self.epsilon = self.epsilon.min
    
    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_targ.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_targ.load_checkpoint()

    def learn(self):
        """ Check if the memory counter is less than the batch size, if it is we don't train """
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()

        incidicies = np.arange(self.batch_size)
        # dims - > batch_size x num_actions
        q_pred = self.q_eval.forward(states)[incidicies, actions]
        # dim 1 is the action dimension
        # [0] gives changes named tuples into actions
        q_next = self.q_targ.forward(states_).max(dim=1)[0]

        # we set-up the dones flag as a amask
        q_next[dones] = 0.0
        # as such, if done flag is 1, q_next has already been set to zero
        q_target = rewards + self.gamma*q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter +=1
        
        self.decrement_epsilon()