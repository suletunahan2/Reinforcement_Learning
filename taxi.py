# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 17:31:05 2020

@author: suletunahan2
"""
'''
Pick up the passenger at one location and drop them off in another
Drop off the passenger to the right location.
Save passenger's time by taking minimum time possible to drop off
State:
5x5 grid = 25 Cell
4 locations that we can pick up and drop off a passenger: R, G, Y, B 
The agent encounters one of the 500 states and it takes an action. 5*5*5*4 = 500
Action:
Action in our case can be to move in a direction or decide to pickup/dropoff a passenger.
Action Space: South, north, east, west, pickup, dropoff
There are 6 discrete deterministic actions:
- 0: move south
- 1: move north
- 2: move east 
- 3: move west 
- 4: pickup passenger
- 5: dropoff passenger

Reward:
Positive reward for a successful dropoff (+20)
Negative reward for a wrong dropoff (-10)
Slight negative reward every time-step (-1) 
Each successfull dropoff is the end of an episode

'''
# pip install gym

import gym
import numpy as np
import random
import matplotlib.pyplot as plt 
env = gym.make('Taxi-v3').env #soyut bir yapı kurmak için .env yapılır.
env.render() # render = show
'''blue:passenger ,purple:destination, yellow:empty taxi ,green:full taxi ,RGBY : location for destination and passenger.'''

env.reset() # Restart the environment to start a new episode

print('State space : ',env.observation_space) # 5*5*5*4=500 discrete state
print('Action Space : ',env.action_space) # 6 discere action


state=env.encode(3,1,2,3) #env.encode(taxi_row,taxi_coulumn,passenger_index,destination)
print('State number: ' ,state)
env.s=state
env.render()

'''
alttaki kod icin output:
(PROBABİLİTY , NEXT_STATE , REWARD ,DONE)

probability:deterministic olduğu için 1 dir.

done : episode bitip bitmemesi

*Yolcuyu* doğru yerde bıraktığında true olacak'''

#331. STATE DEKİ FARKLI ACTİON DA NE OLACAĞI
env.P[331] 

print("********************************************************************")

# choose action-->perform action and get reward --> total reward

# Restart the environment to start a new episode
obs = env.reset() # initialize

for step_idx in range(500):
  env.render() #render = show
  obs, reward, done, _ = env.step(env.action_space.sample())

# Q table
q_table = np.zeros([env.observation_space.n,env.action_space.n])

# setting hyperparameters
alpha= 0.1  #learning rate
gamma = 0.9 #discount factor
epsilon = 0.1 #trade-off between exploration and exploitation

# Plotting Metrix
reward_list = []
droputs_list = []

episode_number = 10000
for i in range(1,episode_number):
    
    # initialize enviroment
    state = env.reset()

    
    reward_count = 0
    dropouts = 0
    
    while True:
        
        # exploit vs explore to find action
        # %10 = explore, %90 exploit
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample()# explore action space
        else:
            action = np.argmax(q_table[state])# exploit learned values

        
        # action process and take reward/ observation
        next_state, reward, done, info= env.step(action)
        
        # Q learning function
        old_value = q_table[state,action] # old_value
        next_max = np.max(q_table[next_state]) # next_max
        
        next_value = (1-alpha)*old_value + alpha*(reward + gamma*next_max)
        
        # Q table update 
        q_table[state,action] = next_value
        
        # update state
        state = next_state
        
        # find wrong dropouts
        if reward == -10:
            dropouts += 1
            
        
        if done:
            break
        
        reward_count += reward 

# %% visualize
fig ,axs = plt.subplots(1,2)
axs[0].plot(reward_list)
axs[0].set_xlabel("episode")
axs[0].set_ylabel("reward")

axs[1].plot(droputs_list)
axs[1].set_xlabel("episode")
axs[1].set_ylabel("dropouts")

axs[0].grid(True)
axs[1].grid(True)

plt.show()

# taxi row, taxi, column, passenger index, destination 
       
env.s = env.encode(0,0,3,4) 
env.render()   

  
env.s = env.encode(4,4,4,3) 
env.render()     

np.argmax(q_table[454]) #argmax function return the position of the 
#maximum value among those in the vector examined

#As you can see, the argmax function return position 1, which corresponds to action ‘north’. So, for each position, our q-table will tell us which is the action which maximizes current and future rewards.
q_table[328]      






