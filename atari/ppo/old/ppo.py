import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gym
import numpy as np
from utils import plot_learning_curve, make_env

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(device))

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()
        print("State Dimensions: {}".format(state_dim))
        # actor
        self.act_conv1 = nn.Conv2d(state_dim[0], 32, 8, stride=4)
        self.act_conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.act_conv3 = nn.Conv2d(64, 64, 3, stride=1)
        fc_input_dims = self.act_calculate_conv_output_dims(state_dim)
        self.act_fc1 = nn.Linear(fc_input_dims, 512)
        self.act_fc2 = nn.Linear(512, 300)
        self.act_fc3 = nn.Linear(300, action_dim)
        # critic
        self.crt_conv1 = nn.Conv2d(state_dim[0], 32, 8, stride=4)
        self.crt_conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.crt_conv3 = nn.Conv2d(64, 64, 3, stride=1)
        fc_input_dims = self.crt_calculate_conv_output_dims(state_dim)
        self.crt_fc1 = nn.Linear(fc_input_dims, 512)
        self.crt_fc2 = nn.Linear(512, 300)
        self.crt_fc3 = nn.Linear(300, 1)

    def act_calculate_conv_output_dims(self, input_dimensions):
        state = torch.zeros(1, *input_dimensions)
        dims = self.act_conv1(state)
        dims = self.act_conv2(dims)
        dims = self.act_conv3(dims)
        return int(np.prod(dims.size()))

    def crt_calculate_conv_output_dims(self, input_dimensions):
        state = torch.zeros(1, *input_dimensions)
        dims = self.crt_conv1(state)
        dims = self.crt_conv2(dims)
        dims = self.crt_conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self):
        raise NotImplementedError
        
    def getActionProbs(self, state):
        conv1 = F.relu(self.act_conv1(state))
        conv2 = F.relu(self.act_conv2(conv1))
        conv3 = F.relu(self.act_conv3(conv2))
        # conv3 shape is BS x num_filters x H x W
        # we want to flatten this before passing
        # into the fully connected layers
        conv_state = conv3.view(conv3.size()[0], -1)
        act_fc1 = F.relu(self.act_fc1(conv_state))
        act_fc2 = self.act_fc2(act_fc1)
        actions = self.act_fc3(act_fc2)
        #print(actions.shape)
        action_probs = F.softmax(actions, dim=1)
        return action_probs

    def act(self, state, memory):
        rs_state = np.expand_dims(state, axis=0)
        rs_state = torch.from_numpy(rs_state).float().to(device) 
        action_probs = self.getActionProbs(rs_state)
        dist = Categorical(action_probs)
        action = dist.sample()
        memory.states.append(torch.from_numpy(state).float().to(device))
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        return action.item()

    def getStateValue(self, state):
        conv1 = F.relu(self.crt_conv1(state))
        conv2 = F.relu(self.crt_conv2(conv1))
        conv3 = F.relu(self.crt_conv3(conv2))
        # conv3 shape is BS x num_filters x H x W
        # we want to flatten this before passing
        # into the fully connected layers
        conv_state = conv3.view(conv3.size()[0], -1)
        crt_fc1 = F.relu(self.crt_fc1(conv_state))
        crt_fc2 = self.crt_fc2(crt_fc1)
        value = self.crt_fc3(crt_fc2)
        return value
    
    def evaluate(self, state, action):
        action_probs = self.getActionProbs(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.getStateValue(state)
        return action_logprobs, torch.squeeze(state_value), dist_entropy
        
class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def update(self, memory):   
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
                
            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
def main():
    ############## Hyperparameters ##############
    env_name = "PongNoFrameskip-v4"
    # creating environment
    #env = gym.make(env_name)
    env = make_env(env_name=env_name, repeat=4, clip_reward=False,
                    no_ops=0, fire_first=False)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    render = False
    solved_reward = 230         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 50000        # max training episodes
    max_timesteps = 300         # max timesteps in one episode
    n_latent_var = 64           # number of variables in hidden layer
    update_timestep = 2000      # update policy every n timesteps
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    random_seed = None
    #############################################
    
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    print(lr,betas)
    
    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    
    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(max_timesteps):
            timestep += 1
            
            # Running policy_old:
            action = ppo.policy_old.act(state, memory)
            state, reward, done, _ = env.step(action)
            
            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            # update if its time
            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0
            
            running_reward += reward
            if render:
                env.render()
            if done:
                break
                
        avg_length += t
        
        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("Solved!")
            torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))
            break
            
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))
            
            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
            
if __name__ == '__main__':
    main()
    