#ray==0.8.7 에서 실행
#강화 학습 알고리즘인 policy gradient를 활용하여 유명한 atari game 환경에서 학습을 진행해본다.
#학습 환경은 openai gym에서 제공하는 'BreakoutNoFrameskip-v4'로 설정

import ray
import ray.rllib.agents.pg as pg
from ray.tune.logger import pretty_print

ray.init()
config= {'train_batch_size': 10000,
	'num_gpus':0.8,
        'num_workers': 4,
        'batch_mode': 'complete_episodes',
        'observation_filter': 'MeanStdFilter',
	'monitor': False, 'lr': 0.0004}
trainer = pg.PGTrainer(config=config, env="BreakoutNoFrameskip-v4")  #set environment


for i in range(1000):
   print("iteration:",i)
   # Perform one iteration of training the policy with policy gradient
   result = trainer.train()
   print(pretty_print(result))

   if i % 200 == 0:
       checkpoint = trainer.save('./checkpoint_pg_atari') # set checkpoint save path
       print("checkpoint saved at", checkpoint)
