import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import *


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(FEATURES, HIDDEN_LAYER_SIZE)
        self.fc2 = nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
        self.fc3 = nn.Linear(HIDDEN_LAYER_SIZE, 1)  # In binary classification using BCE loss we return a single
        # value in the output layer.

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)  # Probability of belonging to the positive class
