#python reinforce.py --policy=snn --random_seed=9999 --environment=CartPole-v1 &
#python reinforce.py --policy=ann --random_seed=9999 --environment=CartPole-v1 &
#python actor_critic.py --policy=ann-ac --random_seed=9999 --environment=CartPole-v1 &
#python actor_critic.py --policy=snn-ac --random_seed=9999 --environment=CartPole-v1 &
python ppo_3.py --policy=ann-ppo --random_seed=1234 --epoch=2 --episodes=3000 --environment=CartPole-v1 &
# python ppo_3.py --policy=ann-ppo --random_seed=1234 --epoch=1 --episodes=3000 --environment=CartPole-v1 &
python ppo_3.py --policy=snn-ppo --random_seed=1234 --epoch=2 --episodes=3000 --environment=CartPole-v1 &
# python ppo_3.py --policy=snn-ppo --random_seed=1234 --epoch=1 --episodes=3000 --environment=CartPole-v1 &
