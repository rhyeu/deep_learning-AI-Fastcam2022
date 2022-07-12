#ray==0.8.7 에서 실행
#강화 학습 알고리즘인 actor-critic을 활용하여 유명한 atari game 환경에서 학습을 진행해본다.
#학습 환경은 openai gym에서 제공하는 'BreakoutNoFrameskip-v4'로 설정

import ray
import ray.rllib.agents.a3c.a2c as a2c
from ray.tune.logger import pretty_print

ray.init()

config= {'train_batch_size': 10000,
	'num_gpus':0.8,
        'num_workers': 4,
        'rollout_fragment_length': 20,
        'clip_rewards': True,'num_envs_per_worker': 3,
	'monitor': False, 'lr_schedule': [[0, 0.0007],[20000000, 0.000000000001]]}

trainer = a2c.A2CTrainer(config=config, env="BreakoutNoFrameskip-v4")  #set environment


for i in range(1000):
   print("iteration:",i)
   # Perform one iteration of training the policy with actor-critic
   result = trainer.train()
   print(pretty_print(result))

   if i % 200 == 0:
       checkpoint = trainer.save('./checkpoint_a2c_atari') # set checkpoint save path
       print("checkpoint saved at", checkpoint)
