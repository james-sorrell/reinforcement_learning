import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, name, input_dims, checkpoint_dir, num_actions, lr):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.createModel(input_dims, num_actions, lr)

    def createModel(self, input_dimensions, num_outputs, lr):
        # input channels, output filters, kernel size, stride
        self.conv1 = nn.Conv2d(input_dimensions[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        fc_input_dims = self.calculate_conv_output_dims(input_dimensions)
        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, num_outputs)
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dimensions):
        state = T.zeros(1, *input_dimensions)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        # conv3 shape is BS x num_filters x H x W
        # we want to flatten this before passing
        # into the fully connected layers
        conv_state = conv3.view(conv3.size()[0], -1)
        fc1 = F.relu(self.fc1(conv_state))
        actions = self.fc2(fc1)
        return actions

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
        print("\t--> Checkpoint saved")

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
        print("\t--> Checkpoint loaded")