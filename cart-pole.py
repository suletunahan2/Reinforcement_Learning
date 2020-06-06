# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 23:48:30 2020

@author: suletunahan2
"""
''' Template:
Agent Class:
init() for decribe agent
ann_model() for find a action (ann : Artificial Neural Network)
remember() for replay memory (input parameters of remember() is s,a,s',r )
act() output is action, input is state
replay() for training 
adaptiveE() for exploration & exploitation


Environment Class:(Gym toolkit)
reset() for initialize
step() for action 
'''
import gym #environment class
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class agentClass:
  #constructor
  def __init__(self,env):
    # define parameter/hyperparameter

    #for model method
    self.state_size=env.observation_space.shape[0] #for ann model input(neuron sayısı)
    self.action_size=env.action_space.n #for ann model output

    #for replay method
    self.gamma=0.95 # future reward
    self.alfa=0.001 #agentın öğrenme hızını belirler.

    #for adaptiveE method
    self.epsilon=1 #baslangıcta her yeri ara
    self.epsilon_decay=0.995 #bu kadar azalt
    self.epsilon_min=0.01

    #for create storage
    self.memory=deque(maxlen=1000) #1000 lik liste gibi düsün.Dolarsa ilk listeye giren atılır.

    #for agent nn
    self.model=self.ann_model()


  def ann_model(self):
    #ANN with keras
    model=Sequential()
    model.add(Dense(48,input_dim=self.state_size,activation="tanh"))
    #dense:hiddenlayer,input_dim:self.state_size
    model.add(Dense(self.action_size,activation="linear"))
    #self.state_size:bu kadar output nueron olacak cunku output action olur.
    model.compile(loss="mse",optimizer=Adam(lr=self.alfa))
    return model
    


  def remember(self,state,action,reward,next_state,done):
    #state,action,reward,next_state,done input parameter for storage 
    self.memory.append((state,action,reward,next_state,done))

  def act(self,state):
    #action (explore-exploit)
    if random.uniform(0,1)<=self.epsilon:
      return env.action_space.sample()
    else:
      #en büyük q value nın indexi actiondır.
      act_values=self.model.predict(state)#q value
      return np.argmax(act_values[0])


  def replay(self,batch_size):#tekrarlama
    #batch_size:rememberdan ne kadar deneyimimi kullanacağımı belirlediğim parametre 
    #training
    if len(self.memory)<batch_size:#memory dolu mu
      return #çık
    minibatch=random.sample(self.memory,batch_size)
    for state, action, reward, next_state, done in minibatch:
      if done:#basarısızsa gelecek olmadıgı icin
        target = reward 
      else:#hala basarısız olmadıysa
        target = reward + self.gamma*np.amax(self.model.predict(next_state)[0])#amax bütün hepsini birleştirip büyüğü bulur listede
      train_target = self.model.predict(state)#trained q values
      train_target[0][action] = target
      self.model.fit(state,train_target, verbose = 0)#back propagation,verbose=0 : print etme




  def adaptiveE(self):
    #exploration & exploitation
    if self.epsilon > self.epsilon_min:
      #self.epsilon = self.epsilon*self.epsilon_decay
      self.epsilon *= self.epsilon_decay

  


  



if __name__ == "__main__":

  #initialize environment and agent
  env=gym.make("CartPole-v0")
  agent=agentClass(env) #agent classından nesne ürettik.
  batch_size=16
  episodes=120
  for e in range(episodes):
      #initialize environment for each episodes
      state=env.reset()
      state=np.reshape(state,[1,4])#[[],[]....] haline getirdik.
      time=0




      while True:
        #select an action(hareketi belirle)
        action=agent.act(state)

        #step(environment de bu hareketi uygula)
        next_state,reward,done,_=env.step(action)
        next_state=np.reshape(state,[1,4])

        #remember-storage (environment a,r,s' return edecek ve bunlar remember da kullanılacak)
        agent.remember(state,action,reward,next_state,done)
  
        #update state
        state=next_state

        #replay(tecrubelerden yararlanma)
        agent.replay(batch_size) # rastgele 16 tane (state,action,reward,next_state,done) kullanacagız demektir.

        #set epsilon(adjust)
        agent.adaptiveE()


        time+=1

        if done: #step methodundan gelen bilgi 
          print(f"Episode:{e} ,time:{time}")
          break
'''
# %% testing
import time

trained_model = agent 
state = env.reset()
state = np.reshape(state, [1,4])
time_t = 0
while True:
    env.render()
    action = trained_model.act(state)
    next_state, reward, done, _ = env.step(action)
    next_state = np.reshape(next_state, [1,4])
    state = next_state
    time_t += 1
    print(time_t)
    #time.sleep(0.4)
    if done:
        break
print("Done")
#time:200 demek %100 basarılı demektir.'''
import matplotlib.pyplot as plt
%matplotlib inline
from IPython import display
def show_state(env, step=0, info=""):
    plt.figure(3)
    plt.clf()
    plt.imshow(env.render(mode='rgb_array'))
    plt.title("%s | Step: %d %s" % (env._spec.id,step, info))
    plt.axis('off')

    display.clear_output(wait=True)
    display.display(plt.gcf())

