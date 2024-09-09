import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete
from utils.networks import MLPNetwork
from utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from utils.agents import DDPGAgent
import numpy as np

MSELoss = torch.nn.MSELoss()

class DDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_init_params, gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64, discrete_action=False):

        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.agent = DDPGAgent(lr=lr, discrete_action=discrete_action, hidden_dim=hidden_dim, **agent_init_params)
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action
        self.pol_dev = 'cpu'  # device for policy
        self.critic_dev = 'cpu'  # device for critic
        self.trgt_pol_dev = 'cpu'  # device for target policy
        self.trgt_critic_dev = 'cpu'  # device for target critic
        self.niter = 0

    @property
    def policy(self):
        return self.agent.policy

    @property
    def target_policy(self):
        return self.agent.target_policy

    def scale_noise(self, scale):
        """
        Scale noise for the agent
        Inputs:
            scale (float): scale of noise
        """
        self.agent.scale_noise(scale)

    def reset_noise(self):
        self.agent.reset_noise()

    def step(self, observation, explore=False, guided=True):
        """
        Take a step forward in environment with the agent
        Inputs:
            observation: Observation for the agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action: Action for the agent
        """
        self.agent.epsilon = max(0.01, self.agent.epsilon - 0.0000005)
        self.agent.guide_prob = max(0.01, self.agent.guide_prob - 0.0000005)

        global_explore = explore and np.random.rand() < self.agent.epsilon

        return self.agent.step(observation, explore=global_explore, guided=guided)

    def update(self, sample, parallel=False, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """

        obs, acs, rews, next_obs, dones = sample
        curr_agent = self.agent

        curr_agent.critic_optimizer.zero_grad()
        """Policy Calculation """
        # Policy Calculation
        if self.discrete_action: # one-hot encode action
            trgt_acs = onehot_from_logits(self.target_policy(next_obs))
        else:
            trgt_acs = self.target_policy(next_obs)

        trgt_obs_in = next_obs
        trgt_acts_in = trgt_acs
        target_value = (rews.view(-1, 1) + self.gamma * curr_agent.target_critic(trgt_obs_in, trgt_acts_in) * (1 - dones.view(-1, 1)))

        curr_obs_in = obs
        curr_acts_in = acs

        actual_value = curr_agent.critic(curr_obs_in, curr_acts_in)
        vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss.backward()
        if parallel:
            average_gradients(curr_agent.critic)
        torch.nn.utils.clip_grad_norm(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        curr_agent.policy_optimizer.zero_grad()

        if self.discrete_action:
            curr_pol_out = curr_agent.policy(obs)
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            curr_pol_out = curr_agent.policy(obs)
            curr_pol_vf_in = curr_pol_out

        curr_obs_in = obs
        curr_acts_in = curr_pol_vf_in

        pol_loss = -curr_agent.critic(curr_obs_in, curr_acts_in).mean()
        pol_loss += (curr_pol_out**2).mean() * 1e-3
        pol_loss.backward()

        if parallel:
            average_gradients(curr_agent.policy)
        torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)
        curr_agent.policy_optimizer.step()
        if logger is not None:
            logger.add_scalars('losses', {'vf_loss': vf_loss, 'pol_loss': pol_loss}, self.niter)


    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for the agent)
        """
        soft_update(self.agent.target_critic, self.agent.critic, self.tau)
        soft_update(self.agent.target_policy, self.agent.policy, self.tau)
        self.niter += 1

    def prep_training(self, device='gpu'):
        self.agent.policy.train()
        self.agent.critic.train()
        self.agent.target_policy.train()
        self.agent.target_critic.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            self.agent.policy = fn(self.agent.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            self.agent.critic = fn(self.agent.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            self.agent.target_policy = fn(self.agent.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            self.agent.target_critic = fn(self.agent.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        self.agent.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            self.agent.policy = fn(self.agent.policy)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of the agent into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict, 'agent_params': self.agent.get_params()}
        torch.save(save_dict, filename)


    @classmethod
    def init_from_env(cls, env, gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64):
        """
        Instantiate instance of this class from environment
        """
        agent_init_params = []
        acsp = env.action_space
        obsp = env.observation_space
        print("SHAPE", obsp)
        num_in_pol = obsp.shape[0]

        if isinstance(acsp, Box):
            discrete_action = False
            get_shape = lambda x: x.shape[0]
        else:  # Discrete
            discrete_action = True
            get_shape = lambda x: x.n
        num_out_pol = get_shape(acsp)
        num_in_critic = obsp.shape[0] + get_shape(acsp)
        obs_in_critic = obsp.shape[0]
        act_in_critic = get_shape(acsp)


        agent_init_params = ({'num_in_pol': num_in_pol, 'num_out_pol': num_out_pol,
                              'obs_in_critic': obs_in_critic, 'act_in_critic': act_in_critic,
                                  'num_in_critic': num_in_critic})
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr, 
                     'hidden_dim': hidden_dim, 'agent_init_params': agent_init_params, 
                     'discrete_action': discrete_action}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance
    
    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        instance.agent.load_params(save_dict['agent_params'])
        return instance