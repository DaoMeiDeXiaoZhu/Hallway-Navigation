import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = int(capacity)
        self.ptr = 0
        self.size = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 定义各个向量的经验池
        self.state_buffer = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.action_buffer = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.reward_buffer = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_state_buffer = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.done_buffer = np.zeros((self.capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.state_buffer[self.ptr] = state
        self.next_state_buffer[self.ptr] = next_state
        self.action_buffer[self.ptr] = action
        self.reward_buffer[self.ptr] = reward
        self.done_buffer[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        batch_state = torch.FloatTensor(self.state_buffer[ind]).to(self.device)
        batch_next_state = torch.FloatTensor(self.next_state_buffer[ind]).to(self.device)
        batch_action = torch.FloatTensor(self.action_buffer[ind]).to(self.device)
        batch_reward = torch.FloatTensor(self.reward_buffer[ind]).to(self.device)
        batch_done = torch.FloatTensor(self.done_buffer[ind]).to(self.device)

        return batch_state, batch_action, batch_reward, batch_next_state, batch_done

    def __len__(self):
        return self.size



