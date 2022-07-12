#ray==0.8.7 에서 실행
#학습을 완료하여 저장한 checkpoint 파일을 로드하여 동일한 환경에서 실행을 시키고 gui로 확인을 해본다.

import ray
from ray.tune.logger import pretty_print
from ray.rllib import agents
import ray.rllib.agents.ppo as ppo
ray.init()
checkpoint_path='./checkpoint_ppo_default_CartPole/checkpoint_802/checkpoint-802'
config = ppo.DEFAULT_CONFIG.copy()
config['monitor']=True #available monitoring through GUI
config['num_workers']=1
config['env']='CartPole-v0' #set environment
trainer = agents.ppo.PPOTrainer(env='CartPole-v0', config=config) 
ray.tune.run(agents.ppo.PPOTrainer, config=config, stop={"training_iteration": 10},restore=checkpoint_path) #stop when iteration is 10

