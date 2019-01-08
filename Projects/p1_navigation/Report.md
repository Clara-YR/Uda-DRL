[//]: # (Image References)

[image1]: img/Fixed_Target.png
[image2]: img/DQN_Architecture.png
[image3]: img/state_shape.png
[image4]: img/DQN.png
[image5]: img/uniform_sample.png
[image6]: img/priority_sample.png
[image7]: img/Dueling_DQN.png

# Report
**Catalog**

- 1. [Deep Q-Learning Algorithm](#1)
	- 1.1 [Q-Learning](#1.1)
	- 1.2 [Experience Replay](#1.2)
	- 1.3 [Fixed Target](#1.3)
- 2. [Neural network](#2)
	- 2.1 [Deep Q-Network](#2.1)
	- 2.2 [My Network Architecture](#2.2)
	- 2.3 [Hyperparameter Tuning](#2.3)
- 3. [Future Improvements](#3)
	- 3.1 [Prioritized Experience Replay](#3.1)
	- 3.2 [Dueling DQN](#3.2)


<a name="1"></a>
## 1. Deep Q-Learning Algorithm

<a name="1.1"></a>
### 1.1 Q-Learning
Q-Learning belongs to temporal-difference(TD) Learning.
TD learninig is a combination of Monte Carlo ideas and dynamic programming (DP) ideas. 

- Like Monte Carlo methods, TD methods can learn directly from raw experience without a model of the environment's dynamics.
- Like DP, TD methods update estimates based in part on other learned estimates, without waiting for a final outcome (they _bootstrap_).

<a name="1.2"></a>
### 1.2 Experience Replay


**class ReplayBuffer**

In both `Navigation.ipynb` and `Navigation_Pixels.ipynb`, I define a class `ReplayBuffer`
to store experience tuples `(state, action, reward, next_state, done)` in SAMPLING step and provided experinece as training dataset in LEARNING step.

<a name="1.3"></a>
### 1.3 Fixed Target
![text][image1]
In Q-Learning, fix the function parameters `W` used to generate target to fixed parameters `Wˉ`.

**class Agent()**

In class `Agent()`, `W` and `Wˉ` are corresponding weights of `qnetwork_local` and `qnetwork_target` respectively. In orther words, `qnetwork_local` has totally same architecture. The only difference is their parameters.
 
- in SAMPLING step:
    - use `qnetwork_local` to calculate Q values and then choose the `action` with the max Q value.
- in LEARNING step:
    - use `qnetwork_target` to calculate TD Target.
    - use `qnetwork_local` to calculate current value
    - update `qnetwork_local` with TD error.
    - copy updated `qnetwork_local` to `qnetwork_target` for the next LEARNING step.
    
In a word, fixed `qnetwork_target` is only used in LEARNING step to make TD Target with constant values while updating our DQN via updating `qnetwork_local`. 

<a name="2"></a>
## 2. Neural network

<a name="2.1"></a>
### 2.1 Deep Q-Network
Tranditional Q-Learning produce only one Q value at a time while Deep Q network can produce a Q value for every possible action in a single forward pass. Thus with DQN, there's no need to run the network individually for every action any more. 

DQN is shown as below:
![text][image4]
It takes states as input, trains the input with deep learning neural network, and then output a _action vector_ containing Q values for all possible actions.


<a name="2.2"></a>
### 2.2 My Network Architecture

**Navigation.ipynb**

Define a class to create Q-Learning Network. My DQN architecture shown in the table below:

|Layer|Input|Output|
|:---:|:---:|:----:|
|fc1|37|500|
|fc2|500|200|
|fc3|200|40|
|fc4|40|4|

Since each input state is a vector instead of a matric, I just implement 4 Fully-Connected Layers to construct my DQN without considering Convolutional Layers. 

**Navigation_Pixels.ipynb**

States shape is shown as below:
![text][image3]
Here state is a matrix instead of a vector, Convolutional Layers should be added to my netwokr. Thus I implement DQN Architecture shown as below to train my model:
![text][image2]

|Layer|Input|Output|Details|
|:---:|:---:|:----:|:-----:|
|Conv1+ReLU|3x84x84|10x40x40|Kernel=5x5, stride=2|
|Conv2+ReLU|10x40x40|10x18x18|Kernel=5x5, stride=2|
|flatten|10x18x18|3240|transform matrix to vector|
|fc1+ReLU|3240|100||
|fc2|100|4||



<a name="2.3"></a>
### 2.3 Hyperparameter Tuning
```
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discout factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY =  4       # how often to update the network
```

- `BUFFER_SIZE`: replay buffer should be big enough to store all experiences.
- `BATCH_SIZE`: I found the total steps of one episode is 300. Set the batch size between 1/5 to 1/4 of the total steps.
- `GAMMA`: for continues space, the influence of follow-up states and actions is obvious to the current state and action, so set the discout factor close to 1.
- `LR`: I'm going to train my network 2000 episodes, so the learning rate should be tine to avoid overfitting.
- `UPDATE_EVERY`: learn frequently.



<a name="3"></a>
## 3. Future Improvements
Here are two furture improvements to my Deep Q-Learning algorithm.

<a name="3.1"></a>
### 3.1 Prioritized Experience Replay

![text][image5]
Deep Q-Learning samples experience transitions uniformly from a replay memory. 

Prioritized experienced replay is based on the idea that the agent can learn more effectively from some transitions than from others, and the more important transitions should be sampled with higher probability.
![text][image6]
The Q value we learn will be biased according to these __priority__ values which we only wanted to use for sampling.
__importance-sampling__ weight is troduce to correct this bias for learning, where N is the size of Replay Buffer.

<a name="3.2"></a>
### 3.2 Dueling DQN
Currently, in order to determine which states are (or are not) valuable, we have to estimate the corresponding action values for each action. 

![text][image7]
However, by replacing the traditional Deep Q-Network (DQN) architecture with a dueling architecture, we can assess the value of each state, without having to learn the effect of each action.