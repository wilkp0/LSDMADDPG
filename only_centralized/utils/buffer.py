import numpy as np
from torch import Tensor
from torch.autograd import Variable


class ReplayBuffer(object):
    """
    Replay Buffer for single-agent RL with parallel rollouts
    """
    def __init__(self, max_steps, obs_dim, ac_dim):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            obs_dim (int): Number of observation dimensions for the agent
            ac_dim (int): Number of action dimensions for the agent
        """
        self.max_steps = max_steps
        self.obs_buff = np.zeros((max_steps, obs_dim))
        self.ac_buff = np.zeros((max_steps, ac_dim))
        self.rew_buff = np.zeros(max_steps)
        self.next_obs_buff = np.zeros((max_steps, obs_dim))
        self.done_buff = np.zeros(max_steps)

        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (overwrite oldest data)

    def __len__(self):
        return self.filled_i

    def push(self, observations, actions, rewards, next_observations, dones):
        nentries = observations.shape[0]  # handle multiple parallel environments
        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i  # num of indices to roll over
            self.obs_buff = np.roll(self.obs_buff, rollover, axis=0)
            self.ac_buff = np.roll(self.ac_buff, rollover, axis=0)
            self.rew_buff = np.roll(self.rew_buff, rollover)
            self.next_obs_buff = np.roll(self.next_obs_buff, rollover, axis=0)
            self.done_buff = np.roll(self.done_buff, rollover)
            self.curr_i = 0
            self.filled_i = self.max_steps

        self.obs_buff[self.curr_i:self.curr_i + nentries] = observations
        self.ac_buff[self.curr_i:self.curr_i + nentries] = actions
        self.rew_buff[self.curr_i:self.curr_i + nentries] = rewards
        self.next_obs_buff[self.curr_i:self.curr_i + nentries] = next_observations
        self.done_buff[self.curr_i:self.curr_i + nentries] = dones

        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

    def sample(self, N, to_gpu=False, norm_rews=True):
        inds = np.random.choice(np.arange(self.filled_i), size=N, replace=False)
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        if norm_rews:
            ret_rews = cast((self.rew_buff[inds] - self.rew_buff[:self.filled_i].mean()) / self.rew_buff[:self.filled_i].std())
        else:
            ret_rews = cast(self.rew_buff[inds])
        return (cast(self.obs_buff[inds]),
                cast(self.ac_buff[inds]),
                ret_rews,
                cast(self.next_obs_buff[inds]),
                cast(self.done_buff[inds]))

    def get_average_rewards(self, N):
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        return self.rew_buff[inds].mean()
