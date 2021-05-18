import os
import random

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from absl import app
from absl import flags
from absl import logging
from norse.torch.functional.lif import LIFParameters
from norse.torch.module.encode import ConstantCurrentLIFEncoder
from norse.torch.module.leaky_integrator import LILinearCell
from norse.torch.module.lif import LIFRecurrentCell

# Hyperparameters
lmbda = 0.95
eps_clip = 0.1
T_horizon = 20

# pytype: enable=import-error

FLAGS = flags.FLAGS
flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to use by pytorch.")
flags.DEFINE_integer("episodes", 1000, "Number of training trials.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate to use.")
flags.DEFINE_float("gamma", 0.99, "discount factor to use")
flags.DEFINE_integer("log_interval", 10, "In which intervals to display learning progress.")
flags.DEFINE_integer("epoch", 3, "Training epochs per episode")
flags.DEFINE_enum("model", "super", ["super"], "Model to use for training.")
flags.DEFINE_enum("policy", "snn-ppo", ["ann-ppo", "snn-ppo"], "Select policy to use.")
flags.DEFINE_boolean("render", False, "Render the environment")
flags.DEFINE_string("environment", "CartPole-v1", "Gym environment to use.")
flags.DEFINE_integer("random_seed", 9999, "Random seed to use")


class ANNPPO(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ANNPPO, self).__init__()
        self.data = []
        self.state_space = kwargs.pop('state_space')
        self.action_space = kwargs.pop('action_space')
        self.fc1 = nn.Linear(self.state_space, 256)
        self.fc_pi = nn.Linear(256, self.action_space)
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=FLAGS.learning_rate)

    def pi(self, x, softmax_dim):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def forward(self, x, model_type, softmax_dim=0):
        if model_type == "v":
            return self.v(x)
        elif model_type == "pi":
            return self.pi(x, softmax_dim=softmax_dim)
        else:
            raise Exception


class SNNPPO(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SNNPPO, self).__init__()
        self.data = []
        self.state_space = kwargs.pop('state_space')
        self.action_space = kwargs.pop('action_space')
        self.constant_current_encoder = ConstantCurrentLIFEncoder(40)
        self.lif = LIFRecurrentCell(
            2 * self.state_space, 256, p=LIFParameters(method="super", alpha=100.0))
        self.readout_pi = LILinearCell(256, self.action_space)
        self.readout_v = LILinearCell(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=FLAGS.learning_rate)

    def pi(self, x, softmax_dim):
        scale = 50
        x_pos = self.constant_current_encoder(torch.nn.functional.relu(scale * x))
        x_neg = self.constant_current_encoder(torch.nn.functional.relu(-scale * x))
        x = torch.cat([x_pos, x_neg], dim=2)
        seq_length, batch_size, _ = x.shape
        voltages_pi = torch.zeros(
            seq_length, batch_size, self.action_space, device=x.device)
        s1 = s0 = None
        for ts in range(seq_length):
            z0, s0 = self.lif(x[ts, :, :], s0)
            vo_pi, s1 = self.readout_pi(z0, s1)
            voltages_pi[ts, :, :] = vo_pi
        out_pi, _ = torch.max(voltages_pi, 0)
        prob = torch.nn.functional.softmax(out_pi, dim=softmax_dim)
        return prob

    def v(self, x):
        scale = 50
        x_pos = self.constant_current_encoder(torch.nn.functional.relu(scale * x))
        x_neg = self.constant_current_encoder(torch.nn.functional.relu(-scale * x))
        x = torch.cat([x_pos, x_neg], dim=2)
        seq_length, batch_size, _ = x.shape
        voltages_v = torch.zeros(seq_length, batch_size, 1, device=x.device)
        s1 = s0 = None
        for ts in range(seq_length):
            z0, s0 = self.lif(x[ts, :, :], s0)
            vo_v, s1 = self.readout_v(z0, s1)
            voltages_v[ts, :, :] = vo_v
        out_v, _ = torch.max(voltages_v, 0)
        return out_v

    def forward(self, x, model_type, softmax_dim=0):
        if model_type == "v":
            return self.v(x)
        elif model_type == "pi":
            return self.pi(x, softmax_dim=softmax_dim)
        else:
            raise Exception


def put_data(network, transition):
    network.data.append(transition)


def make_batch(network):
    s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
    for transition in network.data:
        s, a, r, s_prime, prob_a, done = transition

        s_lst.append(s)
        a_lst.append([a])
        r_lst.append([r])
        s_prime_lst.append(s_prime)
        prob_a_lst.append([prob_a])
        done_mask = 0 if done else 1
        done_lst.append([done_mask])

    s = torch.tensor(s_lst, dtype=torch.float)
    a = torch.tensor(a_lst)
    r = torch.tensor(r_lst)
    s_prime = torch.tensor(s_prime_lst, dtype=torch.float)
    done_mask = torch.tensor(done_lst, dtype=torch.float)
    prob_a = torch.tensor(prob_a_lst)
    network.data = []
    return s, a, r, s_prime, done_mask, prob_a


def train_net(network):
    s, a, r, s_prime, done_mask, prob_a = make_batch(network)

    for i in range(FLAGS.epoch):
        td_target = r + FLAGS.gamma * network(s_prime, model_type='v') * done_mask
        delta = td_target - network(s, model_type='v')
        delta = delta.detach().numpy()

        advantage_lst = []
        advantage = 0.0
        for delta_t in delta[::-1]:
            advantage = FLAGS.gamma * lmbda * advantage + delta_t[0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantage = torch.tensor(advantage_lst, dtype=torch.float)

        pi = network(s, softmax_dim=1, model_type='pi')
        pi_a = pi.gather(1, a)
        ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
        loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(network(s, model_type='v'), td_target.detach())

        network.optimizer.zero_grad()
        loss.mean().backward()
        network.optimizer.step()


def select_action(state, policy, device):
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    probs = policy(state, model_type='pi', softmax_dim=1)
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    return action.item(), probs


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

    env = gym.make(FLAGS.environment)
    env.reset()
    env.seed(FLAGS.random_seed)
    env_state_space = env.observation_space.shape[0]
    env_action_space = env.action_space.n
    model = None  # Variable initialization
    if FLAGS.policy == "ann-ppo":
        model = ANNPPO(state_space=env_state_space, action_space=env_action_space).to(device)
    elif FLAGS.policy == "snn-ppo":
        model = SNNPPO(state_space=env_state_space, action_space=env_action_space).to(device)
    else:
        raise NotImplementedError

    running_rewards = []
    episode_rewards = []

    for n_epi in range(FLAGS.episodes):
        s, ep_reward = env.reset(), 0

        time_steps_max = env._max_episode_steps  # Default was 10000
        for t in range(1, time_steps_max):
            a, prob = select_action(s, model, device)
            s_prime, r, done, info = env.step(a)
            r = float(r)
            if FLAGS.environment == 'CartPole-v1':
                put_data(model, (s, a, r / 100.0, s_prime, prob[0][a].item(), done))
            else:
                put_data(model, (s, a, r, s_prime, prob[0][a].item(), done))
            # put_data(model, (s, a, r, s_prime, prob[0][a].item(), done))
            s = s_prime
            ep_reward += r
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        train_net(model)

        if n_epi % FLAGS.log_interval == 0 and n_epi != 0:
            logging.info(
                "Episode {}/{} \tLast reward: {:.2f}\tAverage reward: {:.2f}".format(
                    n_epi, FLAGS.episodes, ep_reward, running_reward
                )
            )

        episode_rewards.append(ep_reward)
        running_rewards.append(running_reward)

    np.save("running_rewards.npy", np.array(running_rewards))
    np.save("episode_rewards.npy", np.array(episode_rewards))
    torch.save(model.optimizer.state_dict(), "optimizer.pt")
    torch.save(model.state_dict(), "policy.pt")

    env.close()


if __name__ == '__main__':
    app.run(main)
