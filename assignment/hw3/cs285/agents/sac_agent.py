from collections import OrderedDict

from cs285.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.policies.MLP_policy import MLPPolicyAC
from .base_agent import BaseAgent
import gym
from cs285.policies.sac_policy import MLPPolicySAC
from cs285.critics.sac_critic import SACCritic
import cs285.infrastructure.pytorch_util as ptu
import torch
from cs285.infrastructure import sac_utils
class SACAgent(BaseAgent):
    def __init__(self, env: gym.Env, agent_params):
        super(SACAgent, self).__init__()

        self.env = env
        self.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.critic_tau = 0.005
        self.learning_rate = self.agent_params['learning_rate']

        self.actor = MLPPolicySAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
            action_range=self.action_range,
            init_temperature=self.agent_params['init_temperature']
        )
        self.actor_update_frequency = self.agent_params['actor_update_frequency']
        self.critic_target_update_frequency = self.agent_params['critic_target_update_frequency']

        self.critic = SACCritic(self.agent_params)
        self.critic_target = copy.deepcopy(self.critic).to(ptu.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.training_step = 0
        self.replay_buffer = ReplayBuffer(max_size=100000)

    def update_critic(self, ob_no, ac_na, next_ob_no, re_n, terminal_n):
        # TODO: 
        # 1. Compute the target Q value. 
        # HINT: You need to use the entropy term (alpha)
        # 2. Get current Q estimates and calculate critic loss
        # 3. Optimize the critic  
        Q_target, _ = self.critic_target(ob_no[1:], ac_na[1:]).min(dim=1)
        Q_target = torch.cat((Q_target, torch.tensor([0])))
        target = re_n + self.gamma * (1 - terminal_n) * (Q_target - self.actor.alpha())
        Q = self.critic(ob_no, ac_na)
        critic_loss = self.critic.loss(target, Q[0])
        critic_loss += self.critic.loss(target, Q[1])
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()
        return critic_loss

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # TODO 
        # 1. Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic
        critic_loss = 0
        for _ in range(self.agent_params['num_critic_updates_per_agent_update']):
            critic_loss += self.update_critic(ob_no, ac_na, next_ob_no, re_n, terminal_n)
        
        # 2. Softly update the target every critic_target_update_frequency (HINT: look at sac_utils)
        sac_utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)
        
        # 3. Implement following pseudocode:
        # If you need to update actor
        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor
        actor_loss = 0
        alpha_loss = 0
        temperature = 0
        for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
            ac_loss, al_loss, temp = self.actor.update(ob_no, self.critic)
            actor_loss += ac_loss
            alpha_loss += al_loss
            temperature += temp
        # 4. gather losses for logging
        loss = OrderedDict()
        loss['Critic_Loss'] = critic_loss / self.agent_params['num_critic_updates_per_agent_update']# TODO
        loss['Actor_Loss'] = actor_loss / self.agent_params['num_actor_updates_per_agent_update']
        loss['Alpha_Loss'] = alpha_loss / self.agent_params['num_actor_updates_per_agent_update']
        loss['Temperature'] = temperature / self.agent_params['num_actor_updates_per_agent_update']

        return loss

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(batch_size)
