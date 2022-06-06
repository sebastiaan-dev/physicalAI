from collections import deque
import random
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"


class NeuralNetwork(nn.Module):
    def __init__(self, input_state_dim=4, output_state_dim=2):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_state_dim),
        )

    def forward(self, data):
        data = data.to(device)
        return self.linear_relu_stack(data)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward):
        # Add the tuple (state, action, next_state, reward) to the queue
        self.memory.append((state, action, next_state, reward))

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self))
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
