# pytype: disable=import-error
import os
import random
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from absl import app
from absl import flags
from absl import logging

from norse.torch.functional.lif import LIFParameters
from norse.torch.module.encode import ConstantCurrentLIFEncoder
from norse.torch.module.leaky_integrator import LILinearCell
from norse.torch.module.lif import LIFRecurrentCell

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
lmbda = 0.95
eps_clip = 0.1
K_epoch = 3
T_horizon = 20

FLAGS = flags.FLAGS
flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to use by pytorch.")
flags.DEFINE_integer("episodes", 1000, "Number of training trials.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate to use.")
flags.DEFINE_float("gamma", 0.99, "discount factor to use")
flags.DEFINE_integer("log_interval", 10, "In which intervals to display learning progress.")
flags.DEFINE_enum("model", "super", ["super"], "Model to use for training.")
flags.DEFINE_enum("policy", "ann-ppo", ["ann-ppo", "snn-ppo"], "Select policy to use.")
flags.DEFINE_boolean("render", False, "Render the environment")
flags.DEFINE_string("environment", "CartPole-v1", "Gym environment to use.")
flags.DEFINE_integer("random_seed", 9999, "Random seed to use")


class PPO(nn.Module):
    def __init__(self, *args, **kwargs):
        super(PPO, self).__init__()
        self.data = []

        self.state_space = kwargs.pop('state_space')
        self.action_space = kwargs.pop('action_space')
        self.hidden_features = 256

        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=FLAGS.learning_rate)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                              torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                              torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            td_target = r + FLAGS.gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = FLAGS.gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a.to(torch.float)) - torch.log(prob_a.to(torch.float)))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            return loss


class PPOSNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super(PPOSNN, self).__init__()
        self.data = []

        self.state_space = kwargs.pop('state_space')
        self.action_space = kwargs.pop('action_space')
        self.hidden_features = 256
        self.constant_current_encoder = ConstantCurrentLIFEncoder(10)

        self.lif = LIFRecurrentCell(2 * self.state_space, self.hidden_features,
                                    p=LIFParameters(method="super", alpha=100.0))
        self.readout_actor = LILinearCell(self.hidden_features, self.action_space)
        self.readout_critic = LILinearCell(self.hidden_features, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=FLAGS.learning_rate)

    def pi(self, x, softmax_dim=0):
        scale = 50
        x_pos = self.constant_current_encoder(torch.nn.functional.relu(scale * x))
        x_neg = self.constant_current_encoder(torch.nn.functional.relu(-scale * x))
        enc_x = torch.cat([x_pos, x_neg], dim=2)

        seq_length, batch_size, _ = enc_x.shape
        voltages = torch.zeros(seq_length, batch_size, self.action_space, device=enc_x.device)

        s1 = so = None
        # sequential integration loop
        for ts in range(seq_length):
            z1, s1 = self.lif(enc_x[ts, :, :], s1)
            vo, so = self.readout_actor(z1, so)
            voltages[ts, :, :] = vo

        m, _ = torch.max(voltages, 0)
        if True in torch.isnan(m):
            m[m != m] = 0
        p_y = torch.nn.functional.softmax(m, dim=1)

        return p_y

    def v(self, x):
        scale = 50
        x_pos = self.constant_current_encoder(torch.nn.functional.relu(scale * x))
        x_neg = self.constant_current_encoder(torch.nn.functional.relu(-scale * x))
        x = torch.cat([x_pos, x_neg], dim=2)

        seq_length, batch_size, _ = x.shape
        voltages = torch.zeros(seq_length, batch_size, 1, device=x.device)

        s1 = so = None
        # sequential integration loop
        for ts in range(seq_length):
            z1, s1 = self.lif(x[ts, :, :], s1)
            vo, so = self.readout_critic(z1, so)
            voltages[ts, :, :] = vo

        m, _ = torch.max(voltages, 0)
        value = m
        # p_y = torch.nn.functional.softmax(m, dim=1)
        return value

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                              torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                              torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            td_target = r + FLAGS.gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = FLAGS.gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a.to(torch.float)) - torch.log(prob_a.to(torch.float)))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            return loss


def main(args):

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
    env.seed(FLAGS.random_seed)
    env_state_space = env.observation_space.shape[0]
    env_action_space = env.action_space.n

    policy = None  # Variable initialization
    if FLAGS.policy == "ann-ppo":
        policy = PPO(state_space=env_state_space, action_space=env_action_space).to(device)
    elif FLAGS.policy == "snn-ppo":
        policy = PPOSNN(state_space=env_state_space, action_space=env_action_space).to(device)
    else:
        raise NotImplementedError

    running_rewards = []
    episode_rewards = []

    for n_epi in range(FLAGS.episodes):
        s, ep_reward = env.reset(), 0

        for t in range(env._max_episode_steps):
            # prob = policy.pi(torch.from_numpy(s).float())
            prob = policy.pi(torch.from_numpy(s).float().unsqueeze(0).to(device))
            m = Categorical(prob)
            a = m.sample().item()
            s_prime, r, done, info = env.step(a)

            # model.put_data((s, a, r / 100.0, s_prime, prob[a].item(), done))
            policy.put_data((s, a, r, s_prime, a, done))
            s = s_prime

            ep_reward += r
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        policy.train_net()

        if n_epi % FLAGS.log_interval == 0 and n_epi != 0:
            logging.info(
                "Episode {}/{} \tLast reward: {:.2f}\tAverage reward: {:.2f}".format(
                    n_epi, FLAGS.episodes, ep_reward, running_reward
                )
            )

        episode_rewards.append(ep_reward)
        running_rewards.append(running_reward)

    env.close()

    np.save("running_rewards.npy", np.array(running_rewards))
    np.save("episode_rewards.npy", np.array(episode_rewards))

if __name__ == '__main__':
    app.run(main)