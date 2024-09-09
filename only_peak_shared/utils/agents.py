from torch import Tensor
import torch as th
from torch.autograd import Variable
from torch.optim import Adam
from .networks import MLPNetwork, Critic, Actor
from .misc import hard_update, gumbel_softmax, onehot_from_logits
from .noise import OUNoise, Inform
import numpy as np

class DDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """


    def __init__(self, num_in_pol, num_out_pol, obs_in_critic, act_in_critic, num_in_critic,
                  hidden_dim=64, lr=0.01, discrete_action=True):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        print("AGENTS", "*"*10)
        print ("num_in_pol",num_in_pol)
        print ("num_out_pol",num_out_pol)
        print ("obs_in_critic",obs_in_critic)
        print ("act_in_critic",act_in_critic)

        self.policy = Actor(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim)
        self.critic = Critic(obs_in_critic, act_in_critic, 1,
                                 hidden_dim=hidden_dim)
        self.target_policy = Actor(num_in_pol, num_out_pol,
                                        hidden_dim=hidden_dim)
        self.target_critic = Critic(obs_in_critic, act_in_critic, 1,
                                        hidden_dim=hidden_dim)
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        lr_aa = 0.001
        lr_cc = 0.01
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr_aa)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_cc)
        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
            self.epsilon = 0.1  # epsilon for eps-greedy
            self.guide_prob = 0.2  # epsilon for eps-greedy
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action
        self.inform = Inform(num_in_pol,self.exploration)

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False, guided=True):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policy(obs)
        if explore:
            action = self.inform.guided(action, obs)
            #action += Variable(Tensor(self.exploration.noise()),requires_grad=False)
        '''
        # If not 
        if explore and np.random.rand() < self.epsilon:
            action = th.rand_like(action) * 2 - 1  # Assuming action space is [-1, 1]
        else:
            if explore:
                action += Variable(Tensor(self.exploration.noise()),
                                    requires_grad=False)
        '''
        action = action.clamp(-1, 1)
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])



