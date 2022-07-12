#ray==0.8.7 에서 실행
#강화 학습 알고리즘인 ppo를 ray로 학습시켜보자 
#학습 환경은 openai gym에서 제공하는 'CartPole-v0'로 설정

import ray
from ray.tune.logger import pretty_print
from ray.rllib import agents
ray.init ()
config = {'gamma': 0.9,
          'lr': 1e-2,
          'num_workers': 4,
          'train_batch_size': 1000,
          'model': {'fcnet_hiddens': [128, 128]}}
trainer = agents.ppo.PPOTrainer(env='CartPole-v0', config=config) #set environment
results = trainer.train()
for i in range(1000):
   # Perform one iteration of training the policy with PPO
   print("iteration:",i)
   result = trainer.train()
   print(pretty_print(result))
   if i % 200 == 0:
       checkpoint = trainer.save('./checkpoint_ppo_CartPole') # set checkpoint save path
       print("checkpoint saved at", checkpoint)  
