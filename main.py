import argparse, math, os
import numpy as np
import gym
import roboschool
# from gym import wrappers

import torch
from torch.autograd import Variable
import torch.nn.utils as utils

from agent import Agent

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--env_name', type=str, default='RoboschoolInvertedPendulum-v2')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--exploration_end', type=int, default=100, metavar='N',
                    help='number of episodes with noise (default: 100)')
parser.add_argument('--seed', type=int, default=123, metavar='N',
                    help='random seed (default: 123)')
parser.add_argument('--num_steps', type=int, default=1000, metavar='N',
                    help='max episode length (default: 1000)')
parser.add_argument('--num_episodes', type=int, default=2000, metavar='N',
                    help='number of episodes (default: 2000)')

parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='number of hidden neurons (default: 128)')

parser.add_argument('--render', action='store_true',
                    help='render the environment')

parser.add_argument('--ckpt_freq', type=int, default=100,
		    help='model saving frequency')
parser.add_argument('--display', type=bool, default=False,
                    help='display or not')
args = parser.parse_args()

env_name = args.env_name
env = gym.make(env_name)

env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

agent = Agent(args.hidden_size, env.observation_space.shape[0], env.action_space)

for i_episode in range(args.num_episodes):
    state = torch.Tensor([env.reset()])
    log_probs = []
    rewards = []
    for t in range(args.num_steps):
        action, log_prob = agent.select_action(state)
        next_state, reward, done, _ = env.step(action.numpy()[0])
        log_probs.append(log_prob)
        rewards.append(reward)
        state = torch.Tensor([next_state])

        if done:
            break

    agent.update_parameters(rewards, log_probs, args.gamma)

    print("Episode: {}, reward: {}".format(i_episode, np.sum(rewards)))

env.close()
