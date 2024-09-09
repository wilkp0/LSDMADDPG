import argparse
import torch
import time
import imageio
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from utils.make_env_com import make_env
from algorithms.ddpg import DDPG


def run(config):
    model_path = (Path('./models') / config.env_id / config.model_name /
                  ('run%i' % config.run_num))
    if config.incremental is not None:
        model_path = model_path / 'incremental' / ('model_ep%i.pt' % config.incremental)
    else:
        model_path = model_path / 'model.pt'

    ddpg = DDPG.init_from_save(model_path)
    env = make_env(config.env_id, discrete_action=ddpg.discrete_action)
    ddpg.prep_rollouts(device='cpu')

    for ep_i in range(config.n_episodes):
        print("Episode %i of %i" % (ep_i + 1, config.n_episodes))
        obs = env.reset(eval=True)
        for t_i in range(config.episode_length):
            calc_start = time.time()
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = Variable(torch.Tensor(obs), requires_grad=False)
            # get actions as torch Variables
            torch_agent_action = ddpg.step(torch_obs, explore=False)
            # convert actions to numpy arrays
            agent_action = torch_agent_action.data.numpy()
            obs, rewards, dones, infos = env.step(agent_action)
        env.get_final_results()

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="nine", help="Name of environment")
    parser.add_argument("--model_name", default="model_com", help="Name of model")
    parser.add_argument("--run_num", default=2, type=int, help="Run number")
    parser.add_argument("--incremental", default=None, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")
    parser.add_argument("--n_episodes", default=1, type=int)
    parser.add_argument("--episode_length", default=48*5, type=int)

    config = parser.parse_args()

    run(config)