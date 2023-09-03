from typing import Any, Sequence
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pfrl
from pfrl import explorers, replay_buffers

from pfrl.agents import CategoricalDoubleDQN

from pfrl import action_value


from resco_benchmark.agents.agent import IndependentAgent, Agent
from resco_benchmark.config.map_config import map_configs
from torch.distributions import Categorical


def conv2d_size_out(size, kernel_size=2, stride=1):
    return (size - (kernel_size - 1) - 1) // stride + 1


class CategoricalActionValue:
    def __init__(self, distribution):
        self.distribution = distribution

    @property
    def greedy_actions(self):
        return self.distribution.probs.argmax(dim=1)

    def __getitem__(self, index):
        # Extract the logits from the Categorical distribution
        logits = self.distribution.logits
        # Index the logits
        indexed_logits = logits[index]
        # Create a new Categorical distribution with the indexed logits
        new_distribution = torch.distributions.Categorical(logits=indexed_logits)
        return CategoricalActionValue(new_distribution)


class CustomDuelingDQN(nn.Module):
    def __init__(self, n_actions, n_atoms, v_min, v_max, obs_space):
        super(CustomDuelingDQN, self).__init__()

        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.z_values = torch.linspace(v_min, v_max, n_atoms, dtype=torch.float32)

        self.h = conv2d_size_out(obs_space[1])
        self.w = conv2d_size_out(obs_space[2])

        # Convolutional layers
        self.conv = nn.Conv2d(obs_space[0], 64, kernel_size=(2, 2))
        self.activation = nn.ReLU()

        self.main_stream = nn.Linear(self.h * self.w * 64, 128)

        # Dueling branches
        self.a_stream = nn.Linear(64, n_actions * n_atoms)
        self.v_stream = nn.Linear(64, n_atoms)

    def forward(self, x):
        h = self.activation(self.conv(x))
        h = h.view(h.size(0), -1)  # Flatten
        #print(h.size())
        h = self.activation(self.main_stream(h))
        h_a, h_v = torch.chunk(h, 2, dim=1)

        # Advantage
        ya = self.a_stream(h_a).reshape((-1, self.n_actions, self.n_atoms))
        mean = ya.sum(dim=1, keepdim=True) / self.n_actions
        ya, mean = torch.broadcast_tensors(ya, mean)
        ya -= mean

        # State value
        ys = self.v_stream(h_v).reshape((-1, 1, self.n_atoms))
        ya, ys = torch.broadcast_tensors(ya, ys)
        q = F.softmax(ya + ys, dim=2)

        self.z_values = self.z_values.to(x.device)
        return action_value.DistributionalDiscreteActionValue(q, self.z_values)


class RAINBOW(IndependentAgent):
    def __init__(self, config, obs_act, map_name, thread_number):
        super().__init__(config, obs_act, map_name, thread_number)
        for key in obs_act:
            obs_space = obs_act[key][0]
            act_space = obs_act[key][1]

            model = CustomDuelingDQN(act_space, 51, -100, 100, obs_space)

            self.agents[key] = RainbowAgent(config, act_space, model, map_name)


class RainbowAgent(Agent):
    def __init__(self, config, act_space, model, map_name):
        super().__init__()

        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters())

        map_config = map_configs[map_name]
        eval_steps = map_config['end_time'] / map_config['step_length']
        replay_buffer = replay_buffers.PrioritizedReplayBuffer(
            10000,
            alpha=0.5,
            beta0=0.4,
            betasteps=eval_steps/config['TARGET_UPDATE']
        )

        explorer = explorers.LinearDecayEpsilonGreedy(
            config['EPS_START'],
            config['EPS_END'],
            config['steps'],
            lambda: np.random.randint(act_space),
        )

        self.agent = CategoricalDoubleDQN(self.model, self.optimizer, replay_buffer,
                                          gamma=config['GAMMA'],
                                          explorer=explorer,
                                          gpu=self.device.index,
                                          minibatch_size=config['BATCH_SIZE'], replay_start_size=config['BATCH_SIZE'],
                                          phi=lambda x: np.asarray(x, dtype=np.float32),
                                          target_update_interval=config['TARGET_UPDATE']
                                          )

    def act(self, observation, valid_acts=None, reverse_valid=None):
        return self.agent.act(observation)

    def observe(self, observation, reward, done, info):
        self.agent.observe(observation, reward, done, False)

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path + '.pt')
