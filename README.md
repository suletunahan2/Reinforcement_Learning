# Reinforcement_Learning


Q-Learning:

<figure>
 <img src="./Readme-img/qEqu.png" width="500" alt="qEqu" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>

<figure>
 <img src="./Readme-img/qEqu2.png" width="500" alt="qEqu2" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>

Q-Learning Algorithm :


<figure>
 <img src="./Readme-img/alg.png" width="500" alt="alg" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>


Taxi Environment:

<figure>
 <img src="./Readme-img/taxi-1.png" width="500" alt="taxi-1" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>

5x5 grid = 25 Cell
4 locations that we can pick up and drop off a passenger: R, G, Y, B 
The agent encounters one of the 500 states and it takes an action. 5*5*5*4 = 500

Action in our case can be to move in a direction or decide to pickup/dropoff a passenger.
Action Space: South, north, east, west, pickup, dropoff

Positive reward for a successful dropoff (+20)
Negative reward for a wrong dropoff (-10)
Slight negative reward every time-step (-1) 
Each successfull dropoff is the end of an episode


FrozenLake Environment:


<figure>
 <img src="./Readme-img/frozenlake.png" width="500" alt="frozenlake" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>


Deep Q-Learning:

<figure>
 <img src="./Readme-img/deepq.png" width="500" alt="deepqlearning" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>


Deep Q-Learning Algorithm:

<figure>
 <img src="./Readme-img/alg-dql.png" width="500" alt="deepqlearning" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>

Deep Q-Learning Pseudo Code:

<figure>
 <img src="./Readme-img/deepqpseudocode.png" width="500" alt="deepqlearning" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>

CartPole Environment:

<figure>
 <img src="./Readme-img/cart-pole.png" width="800" alt="cartpole" />
 <figcaption>
 <p></p> 
 </figcaption>
</figure>


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





