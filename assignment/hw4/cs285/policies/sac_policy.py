from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
import itertools

class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20,2],
                 action_range=[-1,1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        # TODO: get this from previous HW
        entropy = self.log_alpha.exp()
        return entropy

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # TODO: get this from previous HW
        observations = ptu.from_numpy(obs)
        action_dist = self(observations)
        if sample:
            action = ptu.to_numpy(action_dist.sample())
        else:
            action = action_dist.mean
        return ptu.to_numpy(action)

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: get this from previous HW
        loc = self.mean_net(observation)
        scale = torch.clamp(self.logstd, self.log_std_bounds[0], self.log_std_bounds[1]).exp()
        action_distribution = sac_utils.SquashedNormal(loc, torch.exp(scale))
        return action_distribution

    def update(self, obs, critic):
        # TODO: get this from previous HW

        observation = ptu.from_numpy(obs)
        action_dist = self(observation)
        action = action_dist.sample()
        log_pi = action_dist.log_prob(action)

        Q = critic(observation, ptu.from_numpy(action))
        alpha_log_pi = self.alpha.exp() * log_pi
        actor_loss = (alpha_log_pi - Q).mean()
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        alpha_loss = (-alpha_log_pi - self.alpha.exp() * self.target_entropy).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return actor_loss, alpha_loss, self.alpha