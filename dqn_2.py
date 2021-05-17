# Parts of this code were adapted from the pytorch example at
# https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py
# https://github.com/seungeunrho/minimalRL/blob/master/dqn.py
# which is licensed under the license found in LICENSE.


import os
import random
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import app
from absl import flags
from absl import logging

from norse.torch.functional.lif import LIFParameters
from norse.torch.module.encode import ConstantCurrentLIFEncoder, PopulationEncoder, PoissonEncoder
from norse.torch.module.leaky_integrator import LILinearCell
from norse.torch.module.lif import LIFRecurrentCell

# pytype: disable=import-error

FLAGS = flags.FLAGS
flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to use by pytorch.")
flags.DEFINE_integer("episodes", 1000, "Number of training trials.")
flags.DEFINE_float("learning_rate_actor", 0.0005, "Learning rate to use.")
flags.DEFINE_float("learning_rate_critic", 0.001, "Learning rate to use.")
flags.DEFINE_float("gamma", 0.99, "discount factor to use")
flags.DEFINE_float("tau", 0.005, "target network soft update rate")
flags.DEFINE_integer("batch_size", 1, "batch size for learning from buffer")
flags.DEFINE_integer("buffer_limit", 10000, "replay buffer size")
flags.DEFINE_integer("memory_use_start", 1000, "Samples to fill memory before learning")
flags.DEFINE_integer("hard_update_rate", 10, "Hard update rate")
flags.DEFINE_integer("offline_learning_rate", 10, "Hard update rate")
flags.DEFINE_integer("log_interval", 10, "In which intervals to display learning progress.")
flags.DEFINE_enum("model", "super", ["super"], "Model to use for training.")
flags.DEFINE_enum("policy", "snn-dqn", ["ann-dqn", "snn-dqn"], "Select policy to use.")
flags.DEFINE_boolean("render", False, "Render the environment")
flags.DEFINE_string("environment", "CartPole-v1", "Gym environment to use.")
flags.DEFINE_integer("random_seed", 9999, "Random seed to use")


class ReplayBuffer:
    def __init__(self, *args, **kwargs):
        self.buffer = deque(maxlen=FLAGS.buffer_limit)
        self.device = kwargs.pop('device')

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        state_tensor = torch.tensor(s_lst, dtype=torch.float, device=torch.device(self.device))
        action_tensor = torch.tensor(a_lst, dtype=torch.int64, device=torch.device(self.device))
        reward_tensor = torch.tensor(r_lst, dtype=torch.float, device=torch.device(self.device))
        next_state_tensor = torch.tensor(s_prime_lst, dtype=torch.float, device=torch.device(self.device))
        done_flag_tensor = torch.tensor(done_mask_lst, dtype=torch.float, device=torch.device(self.device))

        return state_tensor, action_tensor, reward_tensor, next_state_tensor, done_flag_tensor

    def size(self):
        return len(self.buffer)


class ANNQNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ANNQNet, self).__init__()
        self.state_space = kwargs.pop('state_space')
        self.action_space = kwargs.pop('action_space')
        self.fc1 = nn.Linear(self.state_space, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon, device):
        state = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        out = self.forward(state)
        coin = random.random()
        if coin < epsilon:
            return np.random.choice(np.arange(self.action_space))
        else:
            return out.argmax().item()


class SNNQNet(torch.nn.Module):
    """
        SNN policy.

    """

    def __init__(self, *args, **kwargs):
        super(SNNQNet, self).__init__()
        self.state_space = kwargs.pop('state_space')
        self.action_space = kwargs.pop('action_space')
        self.constant_current_encoder = ConstantCurrentLIFEncoder(10)
        # self.poulation_encoder = PopulationEncoder(10)
        # self.poisson_encoder = PoissonEncoder(10)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.lif_1 = LIFRecurrentCell(2 * self.state_space, 128, p=LIFParameters(method="super", alpha=100.0))
        # self.fc2 = nn.Linear(128, 64)
        # self.fc3 = nn.Linear(64, 32)
        self.readout = LILinearCell(128, self.action_space)

    def forward(self, x):
        scale = 50
        x_pos = self.constant_current_encoder(torch.nn.functional.relu(scale * x))
        x_neg = self.constant_current_encoder(torch.nn.functional.relu(-scale * x))
        x = torch.cat([x_pos, x_neg], dim=2)

        seq_length, batch_size, _ = x.shape

        voltages = torch.zeros(seq_length, batch_size, self.action_space, device=x.device)

        # s1 = s3 = None
        # # sequential integration loop
        # for ts in range(seq_length):
        #     z1, s1 = self.lif_1(x[ts, :, :], s1)
        #     z2 = F.relu(self.fc2(z1))
        #     z3 = F.relu(self.fc3(z2))
        #     vo, s3 = self.readout(z3, s3)
        #     voltages[ts, :, :] = vo

        s1 = s2 = None
        # sequential integration loop
        for ts in range(seq_length):
            z1, s1 = self.lif_1(x[ts, :, :], s1)
            z1 = self.dropout(z1)
            vo, s2 = self.readout(z1, s2)
            voltages[ts, :, :] = vo

        m, _ = torch.max(voltages, 0)
        q = torch.nn.functional.softmax(m, dim=1)
        return q

    def sample_action(self, obs, epsilon, device):
        state = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        with torch.no_grad():
            out = self.forward(state)
        coin = torch.randn(1)
        if coin < epsilon:
            action = torch.max(out, 1).indices
            return action.item()
        else:
            m = torch.distributions.Categorical(out)
            action = m.sample()
            return action.item()


def train(q, q_target, memory, optimizer, device):
    losses = []
    for i in range(FLAGS.offline_learning_rate):
        s, a, r, s_prime, done_mask = memory.sample(FLAGS.batch_size)
        q_out = q(s)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        q_a = q_out.gather(1, a)
        target = r + FLAGS.gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss)
    loss = sum(losses)/len(losses)
    return loss

def main(args):
    torch.set_default_tensor_type(torch.FloatTensor)
    running_reward = 10
    torch.manual_seed(FLAGS.random_seed)
    random.seed(FLAGS.random_seed)

    label = f"{FLAGS.policy}-{FLAGS.model}-{FLAGS.random_seed}"
    os.makedirs(f"runs/{FLAGS.environment}/{label}", exist_ok=True)
    os.chdir(f"runs/{FLAGS.environment}/{label}")
    if os.path.exists(f"runs/{FLAGS.environment}/{label}/flags.txt"):
        os.remove(f"runs/{FLAGS.environment}/{label}/flags.txt")
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
    env_action_space = env.action_space.n
    policy, policy_target = None, None  # Variable initialization

    if FLAGS.policy == "ann-dqn":
        policy = ANNQNet(state_space=env_state_space, action_space=env_action_space).to(device)
        policy_target = ANNQNet(state_space=env_state_space, action_space=env_action_space).to(device)
        policy_target.load_state_dict(policy.state_dict())
    elif FLAGS.policy == "snn-dqn":
        policy = SNNQNet(state_space=env_state_space, action_space=env_action_space).to(device)
        policy_target = SNNQNet(state_space=env_state_space, action_space=env_action_space).to(device)
        policy_target.load_state_dict(policy.state_dict())
    else:
        raise NotImplementedError

    optimizer = torch.optim.Adam(policy.parameters(), lr=FLAGS.learning_rate_critic)

    running_rewards = []
    episode_rewards = []
    episode_losses = []

    first_seen_flag = True
    episode_loss = 0

    for e in range(FLAGS.episodes):
        epsilon = max(0.01, 0.08 - 0.01 * (e / 200))  # Linear annealing from 8% to 1%
        state, ep_reward = env.reset(), 0

        time_steps_max = env._max_episode_steps  # Default was 10000
        for t in range(1, time_steps_max):
            a = policy.sample_action(state, epsilon, device)
            state_next, reward, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            # memory.put((state, a, reward / 100.0, state_next, done_mask))
            memory.put((state, a, reward, state_next, done_mask))
            state = state_next

            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        if memory.size() > FLAGS.memory_use_start:
            if first_seen_flag is True:
                logging.info(f"Learning started at {e}")
                first_seen_flag = False
            episode_loss = train(policy, policy_target, memory, optimizer, device)

        if e % FLAGS.hard_update_rate == 0:
            policy_target.load_state_dict(policy.state_dict())  # Hard update

        if e % FLAGS.log_interval == 0:
            logging.info("Episode {}/{}\t\tLast reward: {:.2f}\tAverage reward: {:.2f}\tLoss: {:.2f}".format(
                e, FLAGS.episodes, ep_reward, running_reward, episode_loss))

        episode_rewards.append(ep_reward)
        running_rewards.append(running_reward)
        episode_losses.append(episode_loss)

    np.save("running_rewards.npy", np.array(running_rewards))
    np.save("episode_rewards.npy", np.array(episode_rewards))
    np.save("episode_losses.npy", np.array(episode_rewards))

    torch.save(optimizer.state_dict(), "optimizer_critic.pt")
    torch.save(policy.state_dict(), "policy_critic.pt")


if __name__ == "__main__":
    app.run(main)
