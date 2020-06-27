import os
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from models import TD3ActorNetwork, TD3CriticNetwork

class Agent():

    def __init__(self, alpha, beta, input_dims, tau, env,
                gamma=0.99, update_actor_interval=2, warmup=1000,
                n_actions=2, max_size=100000, layer1_size=400, 
                layer2_size=300, batch_size=100, noise=0.1):
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval

        self.critic_1 = TD3CriticNetwork('critic_1', beta, input_dims, layer1_size, layer2_size, n_actions)
        self.critic_2 = TD3CriticNetwork('critic_2', beta, input_dims, layer1_size, layer2_size, n_actions)
        self.actor = TD3ActorNetwork('actor', alpha, input_dims, layer1_size, layer2_size, n_actions)
        self.target_actor = TD3ActorNetwork('target_actor', alpha, input_dims, layer1_size, layer2_size, n_actions)
        self.target_critic_1 = TD3CriticNetwork('target_critic_1', beta, input_dims, layer1_size, layer2_size, n_actions)
        self.target_critic_2 = TD3CriticNetwork('target_critic_2', beta, input_dims, layer1_size, layer2_size, n_actions)

        self.noise = noise
        self.update_network_parameters(tau=1)
    
    def choose_action(self, observation):
        if self.time_step < self.warmup:
            # noise being used as the scale is arbitrary, we are just trying to encourage
            # exploration during the warmup period, random actions
            mu = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions)))
        else:
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise), dtype=T.float).to(self.actor.device)
        mu_prime = T.clamp(mu_prime, self.min_action[0], self.max_action[0])
        self.time_step += 1
        return mu_prime.cpu().detach().numpy()

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)
        
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)
        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        
        target_actions = self.target_actor.forward(state_)
        # Regularisation to help with variance problem associated
        # with function approximators.
        target_actions = target_actions + \
            T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
        # We then need to clamp our actions to be inside our environment bounds
        target_actions = T.clamp(target_actions, self.min_action[0], self.max_action[0])

        q1_ = self.target_critic_1.forward(state_, target_actions)
        q2_ = self.target_critic_2.forward(state_, target_actions)
        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)
        q1_[done] = 0.0
        q2_[done] = 0.0
        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)
        # this is the minimization that td3 suggests
        # in order to reduce overestimation bias
        critic_value_ = T.min(q1_, q2_)
        target = reward + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()

        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        # check if we should continue to update actor network
        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()
        
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        actor_state_dict = dict(actor_params)
        critic_1_state_dict = dict(critic_1_params)
        critic_2_state_dict = dict(critic_2_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_1_state_dict = dict(target_critic_1_params)
        target_critic_2_state_dict = dict(target_critic_2_params)

        for name in critic_1_state_dict:
            critic_1_state_dict[name] = tau * critic_1_state_dict[name].clone() + \
                (1 - tau) * target_critic_1_state_dict[name].clone()

        for name in critic_2_state_dict:
            critic_2_state_dict[name] = tau * critic_2_state_dict[name].clone() + \
                (1 - tau) * target_critic_2_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                (1 - tau) * target_actor_state_dict[name].clone()
        
        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()
