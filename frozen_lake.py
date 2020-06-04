# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 23:40:49 2020

@author: suletunahan2
"""

import gym
from gym import wrappers
import numpy as np
import random
import matplotlib.pyplot as plt #görselleştirme için

env = gym.make('FrozenLake-v0').env

# Restart the environment to start a new episode
obs = env.reset()

for step_idx in range(500):
  env.render()
  obs, reward, done, _ = env.step(env.action_space.sample())
  
print('State space : ',env.observation_space)
print('Action Space : ',env.action_space)

env.observation_space.n #sayıya çevirdik
env.action_space.n

#frozen lake i determinisitc hale getirmek için : is_slippery:false

import time
from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
)

# Q TABLE

q_table =np.zeros([env.observation_space.n,env.action_space.n]) #numpy kütüphanesinden zerolardan oluşan q table oluşturuldu.
#np.zeros([state,action])

# HYPERPARAMETER

alpha=0.8
gamma=0.95
epsilon =0.1 # %10 explore , %90 exploit


reward_list=[]
reward_count=0


# START EPİSODE 
episode_number=10000
for i in range(episode_number):


   #initialize environment
   state= env.reset()

   #agentı bir episode için eğitiriz.yandığımız an biter.

   while True:

     #exploit vs explore to find action(choose an action) (epsilon greedy)
     action=0
     if random.uniform(0,1)<epsilon: # eğer küçükse keşfet
        action=env.action_space.sample() # 6 actiondan rastgele değer al demek
     else:
        action=np.argmax(q_table[state,:])  #np.argmax :en yüksek sayının bulunduğu indexi döndürür.


     #action process and take reward/ observation
     next_state,reward,done,_=env.step(action) #step fonksiyonu action ı gerçekleştirmeye yarayan method


     #Q learning function
     old_value=q_table[state,action]
     next_max= np.max(q_table[next_state])
     next_value=(1-alpha)*(old_value+alpha*(reward+gamma*next_max))


     #Q table update
     q_table[state,action]=next_value

     #update state
     state=next_state
     reward_count+=reward


     if done:  #yanınca çıkmasını sağlayacak
       break
    
   if i%10 == 0:
      reward_list.append(reward_count)
      print("Episode {}, reward {}  ".format(i,reward_count))


plt.plot(reward_list)
q_table
q_table[0]