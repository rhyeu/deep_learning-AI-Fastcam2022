#ray==0.8.7 에서 실행
#강화 학습 알고리즘인 dqn을 ray로 학습시켜본다. 이번에는 Trainer를 직접 사용하여 학습시키는 것이 아니라 tune.run을 활용하여 학습을 해본다. 
#학습 환경은 openai gym에서 제공하는 'CartPole-v0'로 설정
import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer

ray.init()

env_name = 'CartPole-v0'
save_path='./checkpoint_dqn_CartPole' # set checkpoint save path
tune.run(
     DQNTrainer,local_dir=save_path, checkpoint_at_end=True,
     stop={"episode_reward_mean": 195},  # stop until we get target reward
     config={
       "env": env_name,
       "num_workers": 3,
       "num_gpus": 1,
       "monitor": False,
       "evaluation_num_episodes": 25})

