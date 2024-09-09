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
        #need to know which state is which aka. need to know if present

        # Generate Ornstein-Uhlenbeck (OU) noise for the second action component
        ou_noise = th.tensor(self.exploration.noise(), dtype=action.dtype, device=action.device)
        action[0][0] += Variable(ou_noise[0], requires_grad=False)
        #action[0][0] += Variable(th.Tensor(self.exploration.noise()), requires_grad=False)
        #action[0][0] += Variable(ou_noise.squeeze(), requires_grad=False)


        arcsine_noise = self.generate_arcsine_noise(action.shape)
        arcsine_noise = th.tensor(arcsine_noise, dtype=action.dtype, device=action.device)
        action[0][-1] = arcsine_noise[0][0]


        #print("ACTION with Noise", action)
        #print("Noise OU", action[0][0])
        #action = th.rand_like(action) * 2 - 1

        #if obs[0][-1] == 0:
            #force charge to 0
        #    action[0][0] = random.choice([-1, 1])
        #     action[0][0] =  -1 #equal to zero charge when normlaized holy fuck in tarded
        #    pass
        #else:
            #if np.random.rand() < 0.5:
                # Take a random action in the range [-1, 1]
            #    action = th.rand_like(action) * 2 - 1
            #else:
                #action += Variable(th.Tensor(self.exploration.noise()), requires_grad=False)
                # Add OU noise to the action for exploration
        return action
