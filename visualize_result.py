import matplotlib.pyplot as plt
import numpy as np

colors = [['C1', 'C2'], ['C3', 'C4'], ['C5', 'C6'], ['C7', 'C8'], ['C9', 'C10'],['C11', 'C12']]
seeds = ['9999']
env_name = 'CartPole-v1'
main_filepath = f"/home/lab/PycharmProjects/rl-snn-norse/runs/{env_name}/"
filepath_snn = f"snn-super"
filepath_ann = f"ann-super"
filepath_ann_ac = f"ann-ac-super"
filepath_snn_ac = f"snn-ac-super"
# filepath_ann_dqn = f"ann-dqn-super"
filepath_ann_ppo = f"ann-ppo-super"
filepath_snn_ppo = f"snn-ppo-super"
filepaths = [filepath_ann, filepath_snn, filepath_ann_ac, filepath_snn_ac]
filepaths_legend = ['-'.join(i.split('-')[:-1]) for i in filepaths]
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
