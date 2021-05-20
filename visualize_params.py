# Use this command below to search in dqn_params_test
# grep -r . -e 'batch_size=32' -e 'buffer_limit=10000' -e'memory_use_start'

import matplotlib.pyplot as plt
import numpy as np

colors = [['C1', 'C2'], ['C3', 'C4'], ['C5', 'C6'], ['C7', 'C8'], ['C9', 'C10']]
seeds = ['1234']
env_name = 'CartPole-v1'
main_filepath = f"/home/lab/PycharmProjects/rl-snn-norse/runs/{env_name}/ppo_param_test/"
# filepath_test = [f"ann-dqn-super"]
# test_versions = ['1', '6', '7', '8']
# filepaths = ['-'.join(i) for i in zip(filepath_test*len(test_versions), test_versions)]
filepaths = ['ann-ppo-epoch10-super', 'ann-ppo-epoch3-super', 'snn-ppo-epoch10-super', 'snn-ppo-epoch3-super']
filepaths_legend = filepaths
# filepaths_legend = ['-'.join(i.split('-')[:-2]) for i in filepaths]
file_ep_rew = "episode_rewards.npy"
file_avg_rew = "running_rewards.npy"

j = 0
for seed in seeds:
    fig, ax = plt.subplots()
    for i in range(len(filepaths)):
        filepath = f"{main_filepath}{filepaths[i]}-{seed}/"
        y_1 = np.load(filepath + file_ep_rew, fix_imports=True)
        y_2 = np.load(filepath + file_avg_rew, fix_imports=True)
        x = np.arange(1, y_1.size + 1)
        ax.plot(x, y_1, color=colors[i][0], alpha=0.3)
        ax.plot(x, y_2, color=colors[i][1], label=f"{filepaths_legend[i]}")
    ax.set_ylabel('Reward')
    ax.set_title(f'Env: {env_name} - Seed: {seed}')
    plt.legend(loc="lower right")
    fig.show()
