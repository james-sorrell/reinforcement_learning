import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, gamma=0.99, lr=10e-3, update_rate=5*10e-3, batch_size=100):
        self.gamma = gamma
        self.lr = lr
        self.update_rate = update_rate
        self.batch_size = batch_size
        
