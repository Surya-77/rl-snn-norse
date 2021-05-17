from collections import namedtuple
from time import sleep
import numpy as np
import gym
import torch
from absl import app
from absl import flags
from absl import logging
from norse.torch.functional.lif import LIFParameters
from norse.torch.module.encode import ConstantCurrentLIFEncoder
from norse.torch.module.leaky_integrator import LICell
from norse.torch.module.lif import LIFCell
from norse.torch.module.lsnn import LSNNCell, LSNNParameters
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class ActorCriticPolicy(torch.nn.Module):
    """
        Typical ANN policy with state and action space for cartpole defined. The 2 layer network is fully connected
        and has 128 neurons per layer. Uses ReLu activation and softmax final activation.
        Implements both actor and critic in one model
    """

    def __init__(self, *args, **kwargs):
        super(ActorCriticPolicy, self).__init__()

        self.state_space = kwargs.pop('state_space')
        self.action_space = kwargs.pop('action_space')

        self.l1 = torch.nn.Linear(self.state_space, 128, bias=False)
        # actor's layer
        self.l2_actor = torch.nn.Linear(128, self.action_space, bias=False)
        # critic's layer
        self.l2_critic = torch.nn.Linear(128, 1, bias=False)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = self.l1(x)
        x = torch.nn.functional.relu(x)

        # actor: chooses action to take from probability of each action
        action_prob = torch.nn.functional.softmax(self.l2_actor(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.l2_critic(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values


class ReinforcePolicy(torch.nn.Module):
    """
        Typical ANN policy with state and action space for cartpole defined. The 2 layer network is fully connected
        and has 128 neurons per layer. Uses ReLu activation and softmax final activation.
    """

    def __init__(self, *args, **kwargs):
        super(ReinforcePolicy, self).__init__()
        self.state_space = kwargs.pop('state_space')
        self.action_space = kwargs.pop('action_space')
        self.l1 = torch.nn.Linear(self.state_space, 128, bias=False)
        self.l2 = torch.nn.Linear(128, self.action_space, bias=False)
        self.dropout = torch.nn.Dropout(p=0.6)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.l1(x)
        x = self.dropout(x)
        x = torch.nn.functional.relu(x)
        x = self.l2(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x


class ReinforceSNNPolicy(torch.nn.Module):
    """
        SNN policy.

    """

    def __init__(self, *args, **kwargs):
        super(ReinforceSNNPolicy, self).__init__()
        self.state_dim = kwargs.pop('state_space')
        self.input_features = 16
        self.hidden_features = 128
        self.output_features = kwargs.pop('action_space')
        self.constant_current_encoder = ConstantCurrentLIFEncoder(40)
        self.lif = LIFCell(
            2 * self.state_dim,
            self.hidden_features,
            p=LIFParameters(method="super", alpha=100.0),
        )
        self.dropout = torch.nn.Dropout(p=0.5)
        self.readout = LICell(self.hidden_features, self.output_features)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        scale = 50
        x_pos = self.constant_current_encoder(torch.nn.functional.relu(scale * x))
        x_neg = self.constant_current_encoder(torch.nn.functional.relu(-scale * x))
        x = torch.cat([x_pos, x_neg], dim=2)

        seq_length, batch_size, _ = x.shape

        voltages = torch.zeros(
            seq_length, batch_size, self.output_features, device=x.device
        )

        s1 = so = None
        # sequential integration loop
        for ts in range(seq_length):
            z1, s1 = self.lif(x[ts, :, :], s1)
            z1 = self.dropout(z1)
            vo, so = self.readout(z1, so)
            voltages[ts, :, :] = vo

        m, _ = torch.max(voltages, 0)
        p_y = torch.nn.functional.softmax(m, dim=1)
        return p_y


class ActorCriticSNNPolicy(torch.nn.Module):
    """
        SNN policy.

    """

    def __init__(self, *args, **kwargs):
        super(ActorCriticSNNPolicy, self).__init__()
        self.state_dim = kwargs.pop('state_space')
        self.output_features = kwargs.pop('action_space')
        self.input_features = 16
        self.hidden_features = 128
        self.constant_current_encoder = ConstantCurrentLIFEncoder(40)
        self.lif = LIFCell(2 * self.state_dim, self.hidden_features, p=LIFParameters(method="super", alpha=100.0))
        self.readout_actor = LICell(self.hidden_features, self.output_features)
        self.readout_critic = LICell(self.hidden_features, 1)
        self.saved_actions = []
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        scale = 50
        x_pos = self.constant_current_encoder(torch.nn.functional.relu(scale * x))
        x_neg = self.constant_current_encoder(torch.nn.functional.relu(-scale * x))
        x = torch.cat([x_pos, x_neg], dim=2)

        seq_length, batch_size, _ = x.shape

        voltages_actor = torch.zeros(seq_length, batch_size, self.output_features, device=x.device)
        voltages_critic = torch.zeros(seq_length, batch_size, 1, device=x.device)
        s2 = s1 = s0 = None
        # sequential integration loop
        for ts in range(seq_length):
            z0, s0 = self.lif(x[ts, :, :], s0)
            vo_actor, s1 = self.readout_actor(z0, s1)
            vo_critic, s2 = self.readout_critic(z0, s2)
            voltages_actor[ts, :, :] = vo_actor
            voltages_critic[ts, :, :] = vo_critic

        m_actor, _ = torch.max(voltages_actor, 0)
        m_critic, _ = torch.max(voltages_critic, 0)
        action_prob = torch.nn.functional.softmax(m_actor, dim=1)
        state_value = m_critic
        return action_prob, state_value


def select_action(state, policy, device):
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    state_value = None # To stop warning.
    if isinstance(policy, ActorCriticPolicy) or isinstance(policy, ActorCriticSNNPolicy):
        probs, state_value = policy(state)
    else:
        probs = policy(state)
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    if isinstance(policy, ActorCriticPolicy) or isinstance(policy, ActorCriticSNNPolicy):
        policy.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    else:
        policy.saved_log_probs.append(m.log_prob(action))

    return action.item()


episodes = 100
environment = f"Pong-ram-v4"
network_type = f"ann-super"
random_seed = 1234

env = gym.make(environment)
env.reset()
env.seed(random_seed)
env_state_space = env.observation_space.shape[0]
env_action_space = env.action_space.n

model_path = f"/home/lab/PycharmProjects/norse_snn_impl/runs/{environment}/{network_type}-{random_seed}/policy.pt"
policy = None
if network_type == 'ann-super':
    policy = ReinforcePolicy(state_space=env_state_space, action_space=env_action_space).to("cpu")
elif network_type == 'ann-ac-super':
    policy = ActorCriticPolicy(state_space=env_state_space, action_space=env_action_space).to("cpu")
elif network_type == 'snn-ac-super':
    policy = ActorCriticSNNPolicy(state_space=env_state_space, action_space=env_action_space).to("cpu")
elif network_type == 'ann-super':
    policy = ReinforceSNNPolicy(state_space=env_state_space, action_space=env_action_space).to("cpu")
policy.load_state_dict(torch.load(model_path))
policy.eval() # Use eval mode when rendering


device = torch.device("cpu")
for e in range(episodes):
    state, ep_reward = env.reset(), 0
    time_steps_max = 10000  # Default was 10000
    for t in range(1, time_steps_max):  # Don't infinite loop while learning
        action = select_action(state, policy, device="cpu")
        state, reward, done, _ = env.step(action)
        env.render()
        # sleep(0.1)
        policy.rewards.append(reward)
        ep_reward += reward
        if done:
            break
