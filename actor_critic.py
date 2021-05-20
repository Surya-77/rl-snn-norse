# Parts of this code were adapted from the pytorch example at
# https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py
# which is licensed under the license found in LICENSE.

import os
import random
from collections import namedtuple

# pytype: disable=import-error
import gym
import numpy as np
import torch
from absl import app
from absl import flags
from absl import logging
from norse.torch.functional.lif import LIFParameters
from norse.torch.module.encode import ConstantCurrentLIFEncoder
from norse.torch.module.leaky_integrator import LILinearCell
from norse.torch.module.lif import LIFRecurrentCell

# pytype: enable=import-error

FLAGS = flags.FLAGS
flags.DEFINE_enum("device", "cpu", ["cpu", "cuda"], "Device to use by pytorch.")
flags.DEFINE_integer("episodes", 1000, "Number of training trials.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate to use.")
flags.DEFINE_float("gamma", 0.99, "discount factor to use")
flags.DEFINE_integer("log_interval", 10, "In which intervals to display learning progress.")
flags.DEFINE_enum("model", "super", ["super"], "Model to use for training.")
flags.DEFINE_enum("policy", "snn-ac", ["ann-ac", "snn-ac"], "Select policy to use.")
flags.DEFINE_boolean("render", False, "Render the environment")
flags.DEFINE_string("environment", "CartPole-v1", "Gym environment to use.")
flags.DEFINE_integer("random_seed", 9998, "Random seed to use")


class ANNPolicy(torch.nn.Module):
    """
        Typical ANN policy with state and action space for cartpole defined. The 2 layer network is fully connected
        and has 128 neurons per layer. Uses ReLu activation and softmax final activation.
        Implements both actor and critic in one model
    """

    def __init__(self, *args, **kwargs):
        super(ANNPolicy, self).__init__()

        self.state_space = kwargs.pop('state_space')
        self.action_space = kwargs.pop('action_space')

        self.l1 = torch.nn.Linear(self.state_space, 128, bias=False)
        self.l2_actor = torch.nn.Linear(128, self.action_space, bias=False)
        self.l2_critic = torch.nn.Linear(128, 1, bias=False)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = self.l1(x)
        x = torch.nn.functional.relu(x)

        action_prob = torch.nn.functional.softmax(self.l2_actor(x), dim=-1)
        state_value = self.l2_critic(x)

        return action_prob, state_value


class SNNPolicy(torch.nn.Module):
    """
        SNN policy.

    """

    def __init__(self, *args, **kwargs):
        super(SNNPolicy, self).__init__()
        self.state_dim = kwargs.pop('state_space')
        self.output_features = kwargs.pop('action_space')
        self.input_features = 16
        self.hidden_features = 128
        self.constant_current_encoder = ConstantCurrentLIFEncoder(40)
        self.lif = LIFRecurrentCell(2 * self.state_dim, self.hidden_features, p=LIFParameters(method="super", alpha=100.0))
        self.readout_actor = LILinearCell(self.hidden_features, self.output_features)
        self.readout_critic = LILinearCell(self.hidden_features, 1)
        self.saved_actions = []
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
    probs, state_value = policy(state)
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    policy.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item(), probs


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

    env = gym.make(FLAGS.environment)
    env.reset()
    env.seed(FLAGS.random_seed)
    env_state_space = env.observation_space.shape[0]
    env_action_space = env.action_space.n
    policy = None  # Variable initialization
    if FLAGS.policy == "ann-ac":
        policy = ANNPolicy(state_space=env_state_space, action_space=env_action_space).to(device)
    elif FLAGS.policy == "snn-ac":
        policy = SNNPolicy(state_space=env_state_space, action_space=env_action_space).to(device)
    else:
        raise NotImplementedError
    optimizer = torch.optim.Adam(policy.parameters(), lr=FLAGS.learning_rate)

    running_rewards = []
    episode_rewards = []

    for e in range(FLAGS.episodes):
        state, ep_reward = env.reset(), 0

        time_steps_max = env._max_episode_steps  # Default was 10000
        for t in range(1, time_steps_max):  # Don't infinite loop while learning
            action, _ = select_action(state, policy, device=device)
            state, reward, done, _ = env.step(action)
            reward = float(reward)
            if FLAGS.render:
                env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode(policy, optimizer)

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
    torch.save(optimizer.state_dict(), "optimizer.pt")
    torch.save(policy.state_dict(), "policy.pt")


if __name__ == "__main__":
    app.run(main)
