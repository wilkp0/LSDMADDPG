import torch as th
import torch.nn as nn
import torch.nn.functional as F

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=False, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out

    
class Critic(nn.Module):
    def __init__(self, input_dim_observation, input_dim_action, out_dim ,hidden_dim=64, norm_in=False):
        super(Critic, self).__init__()
        if norm_in:
            self.in_fn = nn.BatchNorm1d(input_dim_observation)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
            self.act_fn = nn.BatchNorm1d(input_dim_action)
            self.act_fn.weight.data.fill_(1)
            self.act_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
            self.act_fn = lambda x: x

        self.dim_observation = input_dim_observation
        self.dim_action = input_dim_action
        obs_dim = input_dim_observation
        act_dim = self.dim_action 
        self.out_fn = lambda x: x  # Identity function for the output

        self.FC1 = nn.Linear(obs_dim, hidden_dim)
        self.FC2 = nn.Linear(hidden_dim+act_dim, hidden_dim)
        #self.FC2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC3 = nn.Linear(hidden_dim, out_dim)
        #self.FC4 = nn.Linear(hidden_dim, out_dim)

    # obs: batch_size * obs_dim
    def forward(self, obs, acts):
        obs = self.in_fn(obs)
        acts = self.act_fn(acts)
        result = F.relu(self.FC1(obs))
        combined = th.cat([result, acts], 1)
        result = F.relu(self.FC2(combined))
        return self.out_fn(self.FC3(result))
        #return self.FC4(F.relu(self.FC3(result)))


class Actor(nn.Module):
    def __init__(self, input_dim_observation, out_dim ,hidden_dim=64, norm_in=False):
        super(Actor, self).__init__()
        if norm_in:
            self.in_fn = nn.BatchNorm1d(input_dim_observation)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x

        self.FC1 = nn.Linear(input_dim_observation, hidden_dim)
        self.FC2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC3 = nn.Linear(hidden_dim, out_dim)
        self.FC3.weight.data.uniform_(-3e-3, 3e-3)   

    # action output between -2 and 2
    def forward(self, obs):
        obs = self.in_fn(obs)
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = F.tanh(self.FC3(result))
        return result