
# Deep Q Network for traffic lights control
### (More about model description, see in _Intelligent Transportation System.md_)
[_Deep Reinforcement Learning for Intelligent Transportation Systems_ NIPS 2018 Workshop MLITS](https://openreview.net/forum?id=BJl846ey97)

The states transformation principle is shown in the graph:
<p align="center">
<img src="https://github.com/quantumiracle/DQN_traffic_light_control/blob/master/images/states.png" width="40%">
</p>

__States__ representation:

''0'': green light for direction 1 and hence red light for direction 2;\
''1'': green light for direction 2 and hence red light for direction 1;\
''2'': yellow light for direction 1 and hence red light for direction 2;\
''3'': yellow light for direction 2 and hence red light for direction 1.

__Actions__ representation: (contrary to in paper)

⓪: change state\
①: keep on

## Getting Started

To run this repo, you need to use **Pyhton 3.5**.

## One-way-two-queue intersection

It is one intersection with only two unidirectional roads, no left or right turning. The number of cars on each road is denoted as ![equ.1](https://latex.codecogs.com/gif.latex?Q_1,&space;Q_2\in&space;I) respectively. The state of the traffic light is denoted by state S, which can be in one of the following four states

* "0": green light for road Q_1, and hence red light for road Q_2;
* "1": red light for road Q_1, and hence green light for road Q_1;
* "2": yellow light for Q_1, and red light for road Q_2;
* "3": red light for road Q_1, and yellow light for road Q_2;

And the transition of states, which is called the action in RL algorithm, can be:

* "0": change;
* "1": keep on;

According to general transportation principles, the state transition of traffic lights could only be unidirectional, which is ![equ.1](https://latex.codecogs.com/gif.latex?"0"\rightarrow{"2"}\rightarrow{"1"}\rightarrow{"3"}\rightarrow{"0"}) under our definition of light states above. The trained RL agent takes the tuple [Q_1, Q_2, S] as input and generates action choice for traffic light.
<p align="center">
<img src="https://github.com/quantumiracle/DQN_traffic_light_control/blob/master/images/one_way_two_queue3.png" width="25%">
</p>

The training curve:
<p align="center">
<img src="https://github.com/quantumiracle/DQN_traffic_light_control/blob/master/images/train.png" width="40%">
</p>

Code in **./1.one_way_two_queue/** with `<python light_constr.py --train>` to run.

## Linear-network intersections
Noticing that we don't care much about the outcoming roads, which is denoted by dashed lines.
<p align="center">
<img src="https://github.com/quantumiracle/DQN_traffic_light_control/blob/master/images/linear_network5.png" width="70%">
</p>

States transformation in experiments: the color of lights is 'green' or 'red' or 'yellow'. The black rectangular represents incoming car for periphery of road networks. The numbers indicates number of cars on each road. If the light is 'green', the number of cars in that road will reduce the number of passing cars after transition. If there is 'black rectangular', the number of cars in the correspongding road will increase one after transition. The upper image is the state before transition, while the lower image is the state after transition. 

<p align="center">
<img src="https://github.com/quantumiracle/DQN_traffic_light_control/blob/master/images/Screenshot.png" width="60%">
</p>

The training curve:
<p align="center">
<img src="https://github.com/quantumiracle/DQN_traffic_light_control/blob/master/images/train1.png" width="40%">
</p>

Code in **./2.two_intersections(linear)/** with `<python lights.py --train>` to run.

## Grid-square-network intersections
<p align="center">
<img src="https://github.com/quantumiracle/DQN_traffic_light_control/blob/master/images/grid_square_network.png" width="40%">
 </p>

States transformation in experiments:
  <p align="center">
<img src="https://github.com/quantumiracle/DQN_traffic_light_control/blob/master/images/4*4.png" width="50%">
  </p>

The training curve:
 <p align="center">
<img src="https://github.com/quantumiracle/DQN_traffic_light_control/blob/master/images/2*2_100m.png" width="40%">
</p>

Code in **./3.grid_square_network/** with `<python lights.py --train>` to run.

## Multi-thread version code for grid network

Code in **./4.multithread_for_grid/** with `<python lights.py --train>` to run.

## Agent for single intersection

Code in **./5.one_agent_for_each_intersection/** with `python lights.py --train` or `python lights_re.py --train` to run.

## DDPG version code for grid network

Code in **./6.ddpg_for_grid_network/** with `<python -m run.py --alg=ddpg --num_timesteps=1e4 --train/retrain/test>` to run.
