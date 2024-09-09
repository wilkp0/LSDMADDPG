import numpy as np
from torch import Tensor
import torch as th
from torch.autograd import Variable

# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:
    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale


import random
from scipy.stats import arcsine

# creating an informer that knows which actions to choose.
class Inform:
    def __init__(self, obs_dimension, exploration):
        self.obs_dimension =obs_dimension
        self.exploration = exploration


    def generate_arcsine_noise(self, size):
        # Generate arcsine distributed values in the range [0, 1]
        arcsine_values = arcsine.rvs(size=size)
        # Transform to range [-1, 1]
        transformed_values = 2 * arcsine_values - 1
        return transformed_values
    
    def guided(self, action, obs):
        # Generate Ornstein-Uhlenbeck (OU) noise
        ou_noise = th.tensor(self.exploration.noise(), dtype=action.dtype, device=action.device)

        for i in range(action.shape[1]):
            if i % 2 == 0:
                # Apply OU noise to the even-indexed components
                action[0][i] += Variable(ou_noise[0], requires_grad=False)
            else:
                # Generate and apply arcsine noise for the odd-indexed components
                arcsine_noise = self.generate_arcsine_noise((1, 1))
                arcsine_noise = th.tensor(arcsine_noise, dtype=action.dtype, device=action.device)
                action[0][i] = arcsine_noise[0][0]

        return action