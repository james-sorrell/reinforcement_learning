import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class TD3CriticNetwork(nn.Module):
    def __init__(self, name, beta, input_dims, fc1_dims, fc2_dims, num_actions, checkpoint_dir="tmp/td3"):
        super(TD3CriticNetwork, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.createModel(input_dims, beta, fc1_dims, fc2_dims, num_actions)

    # Designed for one dimensional state space representations
    def createModel(self, input_dims, beta, fc1_dims, fc2_dims, num_actions):
        self.fc1 = nn.Linear(input_dims[0] + num_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q1 = nn.Linear(fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        fc1 = F.relu(self.fc1(T.cat([state, action], dim=1)))
        fc2 = F.relu(self.fc2(fc1))
        q = self.q1(fc2)
        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
        print("\t--> Checkpoint saved")

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
        print("\t--> Checkpoint loaded")


class TD3ActorNetwork(nn.Module):
    def __init__(self, name, alpha, input_dims, fc1_dims, fc2_dims, num_actions, checkpoint_dir="tmp/td3"):
        super(TD3ActorNetwork, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.createModel(input_dims, alpha, fc1_dims, fc2_dims, num_actions)
        
    def createModel(self, input_dims, alpha, fc1_dims, fc2_dims, num_actions):
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        # 4 continuous components of the bipedal walker actions
        self.a1 = nn.Linear(fc2_dims, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        fc1 = F.relu(self.fc1(state))
        fc2 = F.relu(self.fc2(fc1))
        # Here we are activating the output of the action layer
        # binding it to +-1
        a = F.tanh(self.a1(fc2))
        return a

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
        print("\t--> Checkpoint saved")

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
        print("\t--> Checkpoint loaded")