from collections import namedtuple

import gym
import torch
from absl import app

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
import ppo_3
# import actor_critic

episodes = 100
environment = f"CartPole-v1"
arch = "ann"
algo = "ppo"
network_type = f"{arch}-{algo}-super"
random_seed = 9999
log_interval = 10


def main(args):
    running_reward = 10

    env = gym.make(environment)
    env.reset()
    env.seed(random_seed)
    env_state_space = env.observation_space.shape[0]
    env_action_space = env.action_space.n
    model_path = f"/home/lab/PycharmProjects/norse_snn_impl/runs/{environment}/{network_type}-{random_seed}/policy.pt"


    policy = ppo_3.ANNPPO(state_space=env_state_space, action_space=env_action_space)
    select_action = ppo_3.select_action

    policy.load_state_dict(torch.load(model_path))
    policy.eval()  # Use eval mode when rendering
    torch.no_grad()
    device = torch.device("cpu")

    for e in range(episodes):
        state, ep_reward = env.reset(), 0
        time_steps_max = 10000  # Default was 10000
        for t in range(1, time_steps_max):  # Don't infinite loop while learning
            action, _ = select_action(state, policy, device)
            state, reward, done, _ = env.step(action)
            env.render()
            # sleep(0.1)
            ep_reward += reward
            if done:
                break
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        if e % log_interval == 0:
            print("Episode {}/{} \tLast reward: {:.2f}\tAverage reward: {:.2f}"
                  .format(e, episodes, ep_reward, running_reward))


if __name__ == '__main__':
    app.run(main)
