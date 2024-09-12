#ray==0.8.7 에서 실행
#학습을 완료하여 저장한 checkpoint 파일을 로드하여 동일한 환경에서 최종 인퍼런스를 해보고 reward를 구한다.
import ray
import gym
from ray.tune.logger import pretty_print
from ray.rllib import agents
import ray.rllib.agents.ppo as ppo
ray.init ()

config = ppo.DEFAULT_CONFIG.copy() 
agent = agents.ppo.PPOTrainer(env='CartPole-v0', config=config)

checkpoint_path='./checkpoint_ppo_default_CartPole/checkpoint_802/checkpoint-802'

agent.restore(checkpoint_path) #load checkpoint

env = gym.make("CartPole-v0")
#reset environment
obs = env.reset()  
done = False 
episode_reward = 0

#calculate reward 
while not done:
    action = agent.compute_action(obs)
    obs, reward, done, info = env.step(action)
    episode_reward += reward 
    print("episode reward:",episode_reward)
print("Final episode reward:",episode_reward)


