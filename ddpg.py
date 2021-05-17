# Parts of this code were adapted from the pytorch example at
# https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py
# which is licensed under the license found in LICENSE.

import os
import random
from collections import namedtuple, deque

# pytype: disable=import-error
import gym
import numpy as np
import torch
from absl import app
from absl import flags
from absl import logging
from norse.torch.functional.lif import LIFParameters
from norse.torch.module.encode import ConstantCurrentLIFEncoder
from norse.torch.module.leaky_integrator import LICell
from norse.torch.module.lif import LIFCell

# pytype: enable=import-error

FLAGS = flags.FLAGS
flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to use by pytorch.")
flags.DEFINE_integer("episodes", 2000, "Number of training trials.")
flags.DEFINE_float("learning_rate_actor", 0.0005, "Learning rate to use.")
flags.DEFINE_float("learning_rate_critic", 0.001, "Learning rate to use.")
flags.DEFINE_float("gamma", 0.99, "discount factor to use")
flags.DEFINE_float("tau", 0.005, "target network soft update rate")
flags.DEFINE_integer("batch_size", 32, "batch size for learning from buffer")
flags.DEFINE_integer("buffer_limit", 10000, "replay buffer size")
flags.DEFINE_integer("memory_use_start", 2000, "Episodes to fill memory before learning")
flags.DEFINE_integer("update_rate", 10, "Hard update rate")
flags.DEFINE_integer("log_interval", 10, "In which intervals to display learning progress.")
flags.DEFINE_enum("model", "super", ["super"], "Model to use for training.")
flags.DEFINE_enum("policy", "snn-ac-ddpg", ["ann-ac-ddpg", "snn-ac-ddpg"], "Select policy to use.")
flags.DEFINE_boolean("render", False, "Render the environment")
flags.DEFINE_string("environment", "Pendulum-v0", "Gym environment to use.")
flags.DEFINE_integer("random_seed", 9998, "Random seed to use")


class ReplayBuffer():
    def __init__(self, *args, **kwargs):
        self.buffer = deque(maxlen=FLAGS.buffer_limit)
        self.device = kwargs.pop('device')

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_mask_lst.append([done_mask])

        state_tensor = torch.tensor(s_lst, dtype=torch.float, device=torch.device(self.device))
        action_tensor = torch.tensor(a_lst, dtype=torch.float, device=torch.device(self.device))
        reward_tensor = torch.tensor(r_lst, dtype=torch.float, device=torch.device(self.device))
        next_state_tensor = torch.tensor(s_prime_lst, dtype=torch.float, device=torch.device(self.device))
        done_flag_tensor = torch.tensor(done_mask_lst, dtype=torch.float, device=torch.device(self.device))

        return state_tensor, action_tensor, reward_tensor, next_state_tensor, done_flag_tensor

    def size(self):
        return len(self.buffer)


class MuNetANN(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(MuNetANN, self).__init__()

        self.state_space = kwargs.pop('state_space')
        self.action_space = kwargs.pop('action_space')

        self.fc1 = torch.nn.Linear(self.state_space, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc_mu = torch.nn.Linear(64, self.action_space)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))
        return mu


class QNetANN(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(QNetANN, self).__init__()

        self.state_space = kwargs.pop('state_space')
        self.action_space = kwargs.pop('action_space')

        self.fc_s = torch.nn.Linear(self.state_space, 64)
        self.fc_a = torch.nn.Linear(self.action_space, 64)
        self.fc_q = torch.nn.Linear(128, 32)
        self.fc_out = torch.nn.Linear(32, 1)

    def forward(self, x, a):
        h1 = torch.nn.functional.relu(self.fc_s(x))
        h2 = torch.nn.functional.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = torch.nn.functional.relu(self.fc_q(cat))
        q = self.fc_out(q)
        return q


class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


class MuNetSNN(torch.nn.Module):
    """
    2 layer actor network for SNN.
    LIFCell input is doubled for [+ve, -ve] spikes.
    """

    def __init__(self, *args, **kwargs):
        super(MuNetSNN, self).__init__()

        self.state_space = kwargs.pop('state_space')
        self.action_space = kwargs.pop('action_space')

        self.constant_current_encoder = ConstantCurrentLIFEncoder(40)
        self.lif = LIFCell(2 * self.state_space, 128, p=LIFParameters(method="super", alpha=100.0))
        self.readout = LICell(128, self.action_space)

    def forward(self, x):
        scale = 50
        x_pos = self.constant_current_encoder(torch.nn.functional.relu(scale * x))
        x_neg = self.constant_current_encoder(torch.nn.functional.relu(-scale * x))
        x = torch.cat([x_pos, x_neg], dim=2)

        seq_length, batch_size, _ = x.shape

        voltages = torch.zeros(seq_length, batch_size, self.action_space, device=x.device)
        s1 = s0 = None

        # sequential integration loop
        for ts in range(seq_length):
            z0, s0 = self.lif(x[ts, :, :], s0)
            vo, s1 = self.readout(z0, s1)
            voltages[ts, :, :] = vo

        m, _ = torch.max(voltages, 0)
        # action = torch.nn.functional.softmax(m, dim=0)
        action_2 = torch.tanh(m)
        return action_2


class QNetSNN(torch.nn.Module):
    """
        SNN policy.

    """

    def __init__(self, *args, **kwargs):
        super(QNetSNN, self).__init__()

        self.state_space = kwargs.pop('state_space')
        self.action_space = kwargs.pop('action_space')

        self.constant_current_encoder = ConstantCurrentLIFEncoder(40)

        self.lif_s = LIFCell(2 * self.state_space, 64, p=LIFParameters(method="super", alpha=100.0))
        self.lif_a = LIFCell(2 * self.action_space, 64, p=LIFParameters(method="super", alpha=100.0))
        self.lif_c = LIFCell(128, 64, p=LIFParameters(method="super", alpha=100.0))
        self.readout = LICell(64, 1)

    def forward(self, state, action):
        scale = 50

        state_pos = self.constant_current_encoder(torch.nn.functional.relu(scale * state))
        state_neg = self.constant_current_encoder(torch.nn.functional.relu(-scale * state))
        action_pos = self.constant_current_encoder(torch.nn.functional.relu(scale * action))
        action_neg = self.constant_current_encoder(torch.nn.functional.relu(-scale * action))
        state_enc = torch.cat([state_pos, state_neg], dim=2)
        action_enc = torch.cat([action_pos, action_neg], dim=2)

        seq_length_s, batch_size_s, _ = state_enc.shape
        seq_length_a, batch_size_a, _ = action_enc.shape

        voltages = torch.zeros(seq_length_s, batch_size_s, 1, device=state_enc.device)
        # voltages = torch.zeros(seq_length_a, batch_size_a, 1, device=action_enc.device)

        s1 = s0 = None  # Optional state parameters to LIF set to None.

        # sequential integration loop
        for ts in range(seq_length_s):
            z0_s, s0 = self.lif_s(state_enc[ts, :, :], s0)
            z0_a, s0 = self.lif_a(action_enc[ts, :, :], s0)
            com = torch.cat([z0_s, z0_a], dim=1)
            z0, s0 = self.lif_c(com, s0)
            vo, s1 = self.readout(z0, s1)
            voltages[ts, :, :] = vo

        m, _ = torch.max(voltages, 0)
        state_value = m
        return state_value


def select_action(state, policy, device, random_noise=None, scale=2.0):
    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
    result = policy(state_tensor)
    a = result * scale
    if random_noise is not None:
        a = a.item() + random_noise()[0]
    else:
        a = a.item()
    return a


def finish_episode(policy, optimizer):
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    eps = np.finfo(np.float32).eps.item()

    R = 0
    policy_loss = []
    value_loss = []
    returns = []

    for r in policy.rewards[::-1]:
        R = r + FLAGS.gamma * R
        returns.insert(0, R)

    returns = torch.as_tensor(returns)
    returns_amount = len(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    returns.resize_(returns_amount, 1)

    for saved_action, R in zip(policy.saved_actions, returns):
        log_prob, value = saved_action[0], saved_action[1]
        advantage = R - value.item()

        policy_loss.append(-log_prob * advantage)
        value_loss.append(torch.nn.functional.smooth_l1_loss(value, torch.tensor([R])))

    optimizer.zero_grad()

    loss = (torch.stack(policy_loss).sum() + torch.stack(value_loss).sum())
    loss.backward()
    optimizer.step()

    del policy.rewards[:]
    del policy.saved_actions[:]


def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer):
    s, a, r, s_prime, done_mask = memory.sample(FLAGS.batch_size)

    target = r + FLAGS.gamma * q_target(s_prime, mu_target(s_prime)) * done_mask
    q_loss = torch.nn.functional.smooth_l1_loss(q(s, a), target.detach())
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()

    mu_loss = -q(s, mu(s)).mean()
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()


def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - FLAGS.tau) + param.data * FLAGS.tau)


def hard_update(net, net_target):
    net_target.load_state_dict(net.state_dict())


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


def main(args):
    torch.set_default_tensor_type(torch.FloatTensor)
    t = 0
    running_reward = 10
    torch.manual_seed(FLAGS.random_seed)
    random.seed(FLAGS.random_seed)

    label = f"{FLAGS.policy}-{FLAGS.model}-{FLAGS.random_seed}"
    os.makedirs(f"runs/{FLAGS.environment}/{label}", exist_ok=True)
    os.chdir(f"runs/{FLAGS.environment}/{label}")
    FLAGS.append_flags_into_file("flags.txt")

    np.random.seed(FLAGS.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(FLAGS.random_seed)

    device = torch.device(FLAGS.device)

    memory = ReplayBuffer(device=device)

    env = gym.make(FLAGS.environment)
    env.reset()
    env.seed(FLAGS.random_seed)
    env_state_space = env.observation_space.shape[0]
    env_action_space = env.action_space.shape[0]

    if FLAGS.policy == 'ann-ac-ddpg':
        q = QNetANN(state_space=env_state_space, action_space=env_action_space).to(device)
        q_target = QNetANN(state_space=env_state_space, action_space=env_action_space).to(device)
        mu = MuNetANN(state_space=env_state_space, action_space=env_action_space).to(device)
        mu_target = MuNetANN(state_space=env_state_space, action_space=env_action_space).to(device)
    else:
        q = QNetANN(state_space=env_state_space, action_space=env_action_space).to(device)
        q_target = QNetANN(state_space=env_state_space, action_space=env_action_space).to(device)
        mu = MuNetSNN(state_space=env_state_space, action_space=env_action_space).to(device)
        mu_target = MuNetSNN(state_space=env_state_space, action_space=env_action_space).to(device)

    q_target.load_state_dict(q.state_dict())
    mu_target.load_state_dict(mu.state_dict())

    mu_optimizer = torch.optim.Adam(mu.parameters(), lr=FLAGS.learning_rate_actor)
    q_optimizer = torch.optim.Adam(q.parameters(), lr=FLAGS.learning_rate_critic)

    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))

    running_rewards = []
    episode_rewards = []

    for e in range(FLAGS.episodes):
        state, ep_reward = env.reset(), 0

        time_steps_max = 200  # Default was 10000
        for t in range(1, time_steps_max):  # Don't infinite loop while learning
            action = select_action(state, mu, device=device, random_noise=ou_noise)
            state_next, reward, done, _ = env.step([action])
            reward = float(reward)
            memory.put((state, action, reward / 100.0, state_next, done))
            state = state_next
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        if memory.size() > FLAGS.memory_use_start:
            for i in range(FLAGS.update_rate):
                train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer)

        if e % FLAGS.update_rate == 0:
            hard_update(mu, mu_target)
            hard_update(q, q_target)

        if e % FLAGS.log_interval == 0:
            logging.info(
                "Episode {}/{} \tLast reward: {:.2f}\tAverage reward: {:.2f}".format(
                    e, FLAGS.episodes, ep_reward, running_reward
                )
            )
        episode_rewards.append(ep_reward)
        running_rewards.append(running_reward)

    np.save("running_rewards.npy", np.array(running_rewards))
    np.save("episode_rewards.npy", np.array(episode_rewards))
    torch.save(mu_optimizer.state_dict(), "optimizer_actor.pt")
    torch.save(mu.state_dict(), "policy_actor.pt")
    torch.save(q_optimizer.state_dict(), "optimizer_critic.pt")
    torch.save(q.state_dict(), "policy_critic.pt")


if __name__ == "__main__":
    app.run(main)
