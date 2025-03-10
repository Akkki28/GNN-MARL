import gym
import torch
import numpy as np
from gym import spaces
from torch_geometric.data import Data

class GraphPatrolEnv(gym.Env):
    def __init__(self, max_steps=50):
        super().__init__()
        self.num_nodes = 4
        self.edge_index = torch.tensor([
            [0, 1, 2, 3],
            [1, 2, 3, 0]
        ], dtype=torch.long)
        
        self.action_space = spaces.Discrete(2)  
        
        self.max_steps = max_steps
        self.current_step = 0
        self.agent_pos = 0
        self.idleness = np.zeros(self.num_nodes, dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.agent_pos = 0
        self.idleness = np.zeros(self.num_nodes, dtype=np.float32)
        return self._get_observation()

    def step(self, action):
        self.current_step += 1
        neighbors = self._get_neighbors(self.agent_pos)
        next_pos = neighbors[action]
        
        self.idleness += 1.0
        self.idleness[next_pos] = 0.0 
        
        self.agent_pos = next_pos
        
        reward = -float(self.idleness.mean())
        
        done = (self.current_step >= self.max_steps)
        
        obs = self._get_observation()
        return obs, reward, done, {}

    def _get_observation(self):
        x = []
        for i in range(self.num_nodes):
            is_agent = 1.0 if i == self.agent_pos else 0.0
            x.append([self.idleness[i], is_agent])
        
        x = torch.tensor(x, dtype=torch.float32)
        
        data = Data(
            x=x, 
            edge_index=self.edge_index
        )
        return data

    def _get_neighbors(self, node):
        left_neighbor = (node - 1) % self.num_nodes
        right_neighbor = (node + 1) % self.num_nodes
        return [left_neighbor, right_neighbor]

