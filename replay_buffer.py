import numpy as np
from typing import List
import torch


class ReplayBuffer:
    def __init__(self, mem_size, state_dim, action_dim, batch_size):
        self.mem_size = mem_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        assert self.mem_size > self.batch_size, "Batch size cannot be bigger than memory size."
        self.state_mem = np.zeros(shape=(self.mem_size, self.state_dim))
        self.action_mem = np.zeros(shape=(self.mem_size, self.action_dim))
        self.reward_mem = np.zeros(shape=(self.mem_size, 1))
        self.next_state_mem = np.zeros_like(self.state_mem)
        self.terminated_mem = np.zeros(shape=(self.mem_size, 1))
        self.truncated_mem = np.zeros(shape=(self.mem_size, 1))
        self.mem_counter = 0

    def memorize(self, state:List[float], action:float, next_state:List[float],
                 reward:float, term:bool, trunc:bool) -> None:
        # Adds an experience to the memory
        current_counter = self.mem_counter % self.mem_size
        self.state_mem[current_counter] = state
        self.action_mem[current_counter] = action
        self.reward_mem[current_counter] = reward
        self.next_state_mem[current_counter] = next_state
        self.terminated_mem[current_counter] = term
        self.truncated_mem[current_counter] = trunc
        self.mem_counter += 1

    def sample_memories(self):
        #Takes a sample of experiences to use in training
        indices = np.random.randint(low=0, high=self.mem_size, size=self.batch_size)
        states = torch.tensor(self.state_mem[indices, :],
                              dtype=torch.float32).to(device='cuda')
        actions = torch.tensor(self.action_mem[indices, :],
                               dtype=torch.float32).to(device='cuda')
        next_states = torch.tensor(self.next_state_mem[indices, :],
                                   dtype=torch.float32).to(device='cuda')
        rewards = torch.tensor(self.reward_mem[indices, :],
                               dtype=torch.float32).to(device='cuda')
        terms = torch.tensor(self.terminated_mem[indices, :],
                             dtype=torch.float32).to(device='cuda')
        truncs = torch.tensor(self.truncated_mem[indices, :],
                              dtype=torch.float32).to(device='cuda')
        return states, actions, next_states, rewards, terms, truncs
