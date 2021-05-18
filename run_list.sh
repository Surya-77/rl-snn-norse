python reinforce.py --policy=snn --random_seed=9999 --environment=CartPole-v1 &
python reinforce.py --policy=ann --random_seed=9999 --environment=CartPole-v1 &
python actor_critic.py --policy=ann-ac --random_seed=9999 --environment=CartPole-v1 &
python actor_critic.py --policy=snn-ac --random_seed=9999 --environment=CartPole-v1 &
