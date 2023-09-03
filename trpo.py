import torch
import torch.nn as nn
import numpy as np
import pfrl
from resco_benchmark.agents.agent import IndependentAgent, Agent
from torch.distributions import Categorical


def conv2d_size_out(size, kernel_size=2, stride=1):
    return (size - (kernel_size - 1) - 1) // stride + 1


# def observation_phi(obs_dict):
#     """Convert observation dictionary to numpy array."""
#     # Assuming the dictionary has a single key-value pair where the value is the observation
#     key = list(obs_dict.keys())[0]
#     obs = obs_dict[key]
#     return np.asarray(obs, dtype=np.float32)


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class TRPONetwork(nn.Module):
    def __init__(self, obs_space, act_space):
        super(TRPONetwork, self).__init__()

        self.obs_space = obs_space
        self.h = conv2d_size_out(obs_space[1])
        self.w = conv2d_size_out(obs_space[2])

        self.conv = nn.Conv2d(obs_space[0], 64, kernel_size=(2, 2))
        self.fc1 = nn.Linear(64 * self.h * self.w, 64)
        self.fc2 = nn.Linear(64, act_space)

    def forward(self, x):
        x = self.conv(x)
        x = nn.ReLU()(x)
        x = x.view(-1, 64 * self.h * self.w)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        logits = self.fc2(x)
        return Categorical(logits=logits)  # Return a Categorical distribution


class ValueFunction(nn.Module):
    def __init__(self, obs_space):
        super(ValueFunction, self).__init__()
        #self.apply(init_weights)

        self.fc1 = nn.Linear(np.prod(obs_space), 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        return self.fc2(x)


class ITRPO(IndependentAgent):
    def __init__(self, config, obs_act, map_name, thread_number):
        super().__init__(config, obs_act, map_name, thread_number)
        for key in obs_act:
            obs_space = obs_act[key][0]
            act_space = obs_act[key][1]

            policy = TRPONetwork(obs_space, act_space)
            vf = ValueFunction(obs_space)

            self.agents[key] = TRPOAgent(config, act_space, policy, vf)
class TRPOAgent(Agent):
    def __init__(self, config, act_space, policy, vf):
        super().__init__()


        # Define the policy and value function
        self.policy = policy
        self.vf = vf

        # Define the optimizer for the value function
        learning_rate = 0.01
        self.vf_optimizer = torch.optim.Adam(self.vf.parameters(), lr=learning_rate)

        # Initialize the TRPO agent from PFRL
        self.agent = pfrl.agents.TRPO(
            policy=self.policy,
            vf=self.vf,
            vf_optimizer=self.vf_optimizer,
            gpu=self.device.index,
            gamma=config['GAMMA'],
            update_interval=config.get('UPDATE_INTERVAL', 2048),
            conjugate_gradient_max_iter=config.get('CG_MAX_ITER', 10),
            conjugate_gradient_damping=config.get('CG_DAMPING', 1e-1),
            lambd=config.get('LAMBDA', 0.97),
            entropy_coef=config.get('ENTROPY_COEF', 0.0),
            phi=lambda x: np.asarray(x, dtype=np.float32),
        )

    def act(self, observation):
        return self.agent.act(observation)

    def observe(self, observation, reward, done, info):
        self.agent.observe(observation, reward, done, False)

    def save(self, path):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'vf_state_dict': self.vf.state_dict(),
            'optimizer_state_dict': self.vf_optimizer.state_dict(),
        }, path + '.pt')

