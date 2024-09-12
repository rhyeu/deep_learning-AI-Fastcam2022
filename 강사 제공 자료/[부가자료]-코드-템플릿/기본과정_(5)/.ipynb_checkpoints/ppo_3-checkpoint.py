#ray==0.8.7 에서 실행
#강화 학습 알고리즘인 ppo를 ray로 학습시킬때, 학습되는 과정을 gui로 실시간 monitoring이 가능하게 하는법을 배워본다. (ray_results 폴더에 학습되는 과정이 mp4 영상형태로 저장된다.)
#학습 환경은 openai gym에서 제공하는 'CartPole-v0'로 설정

import ray
from ray.tune.logger import pretty_print
from ray.rllib import agents
import ray.rllib.agents.ppo as ppo
ray.init ()
#set config using default value
config = ppo.DEFAULT_CONFIG.copy() 
config['monitor']=True #available monitoring through GUI

trainer = agents.ppo.PPOTrainer(env='CartPole-v0', config=config) #set environment
results = trainer.train()
for i in range(1000):
   # Perform one iteration of training the policy with PPO
   print("iteration:",i)
   result = trainer.train()
   print(pretty_print(result))
   if i % 200 == 0:
       checkpoint = trainer.save('./checkpoint_ppo_default_CartPole') # set checkpoint save path
       print("checkpoint saved at", checkpoint)  


