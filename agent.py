import sys
import math
import pdb

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
import torchvision.transforms as T
from torch.autograd import Variable
from policy import Policy

pi = Variable(torch.FloatTensor([math.pi]))

class Agent:
    def __init__(self, hidden_size, num_inputs, action_space):
        self.action_space = action_space
        self.model = Policy(hidden_size, num_inputs, action_space)
        # self.model = self.model
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.train()

    # probability density of x given a normal distribution
    # defined by mu and sigma
    def normal(self, x, mu, sigma_sq):
        a = (-1*(Variable(x)-mu).pow(2)/(2*sigma_sq)).exp()
        b = 1/(2*sigma_sq*pi.expand_as(sigma_sq)).sqrt()
        return a*b

    def select_action(self, state):
        state = Variable(state)
        mu, sigma_sq = self.model(state)

        # random scalar from normal distribution
        # with mean 0 and std 1
        random_from_normal = torch.randn(1)

        # modulate our normal (mu,sigma) with random_from_normal to pick an action.
        # Note that if x = random_from_normal, then our action is just:
        # mu + sigma * x
        action = (mu + sigma_sq.sqrt()*Variable(random_from_normal)).data

        # calculate the probability density
        prob = self.normal(action, mu, sigma_sq)

        log_prob = prob.log()

        return action, log_prob

    def update_parameters(self, rewards, log_probs, gamma):
        stepReturn = torch.zeros(1, 1)
        loss = 0
        for i in reversed(range(len(rewards))):
            stepReturn = gamma * stepReturn + rewards[i]
            foo = log_probs[i]*Variable(stepReturn)
            loss = loss - foo[0]
        loss = loss / len(rewards)

        self.optimizer.zero_grad()
        loss.backward()
        # utils.clip_grad_norm(self.model.parameters(), 40)
        self.optimizer.step()
