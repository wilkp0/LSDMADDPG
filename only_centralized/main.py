import argparse
import torch
import time
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.ddpg import DDPG

USE_CUDA = False  # torch.cuda.is_available()

def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)
    env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed,
                            config.discrete_action)
    
    print("ENV", env)
    ddpg = DDPG.init_from_env(env, tau=config.tau, lr=config.lr, hidden_dim=config.hidden_dim)

    obs_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0] if isinstance(env.action_space, Box) else env.action_space.n
   
    replay_buffer = ReplayBuffer(config.buffer_length, obs_dim, ac_dim)

    t = 0
    eval_num = 0
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        if ep_i % 100 ==0:
            print("Episodes %i-%i of %i" % (ep_i + 1,
                                            ep_i + 1 + config.n_rollout_threads,
                                            config.n_episodes))
        obs = env.reset(eval=False)
        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
        ddpg.prep_rollouts(device='cpu')

        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        ddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        ddpg.reset_noise()

        for et_i in range(config.episode_length):
            torch_obs = Variable(torch.Tensor(obs), requires_grad=False)
            torch_agent_action = ddpg.step(torch_obs, explore=True)
            agent_action = torch_agent_action.data.numpy()
            next_obs, reward, done, info = env.step(agent_action)
            replay_buffer.push(obs, agent_action, reward, next_obs, done)
            obs = next_obs
            t += config.n_rollout_threads



            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA:
                    ddpg.prep_training(device='gpu')
                else:
                    ddpg.prep_training(device='cpu')
                for u_i in range(config.n_rollout_threads):
                    sample = replay_buffer.sample(config.batch_size, to_gpu=USE_CUDA)
                    ddpg.update(sample, logger=logger)
                    ddpg.update_all_targets()
                ddpg.prep_rollouts(device='cpu')
        ep_rew = replay_buffer.get_average_rewards(config.episode_length* config.n_rollout_threads)
        logger.add_scalar('mean_episode_rewards', ep_rew, ep_i)

        if ep_i % config.eval_interval < config.n_rollout_threads and ep_i != 0:
            eval_rewards = []
            obs = env.reset(eval=True)
            for t_i in range(config.episode_length-1):
                torch_obs = Variable(torch.Tensor(obs), requires_grad=False)
                torch_agent_action = ddpg.step(torch_obs, explore=False)
                agent_action = torch_agent_action.data.numpy()
                obs, reward, done, info = env.step(agent_action)
                eval_rewards.append(reward)
            eval_rews = np.mean(eval_rewards, axis=0)
            for a_i, a_eval_rew in enumerate(eval_rews):
                logger.add_scalar('agent%i/mean_evaluate_rewards' % a_i, a_eval_rew, ep_i)

        if ep_i % (10 * config.eval_interval) == 0:
            env.eval(logger, eval_num)
            eval_num += 1
             
        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            ddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            ddpg.save(run_dir / 'model.pt')
    ddpg.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="com", help="Name of environment")
    parser.add_argument("--model_name", default="model", help="Name of directory to store model/training contents")
    parser.add_argument("--seed", default=1, type=int,help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=9, type=int)
    parser.add_argument("--buffer_length", default=int(5e5), type=int)
    parser.add_argument("--n_episodes", default=260000, type=int)
    parser.add_argument("--episode_length", default=48, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",default=256, type=int,help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=150000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--eval_interval", default=999, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg", default="DDPG", type=str,choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg", default="DDPG", type=str, choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action", action='store_true')

    config = parser.parse_args()

    run(config)
