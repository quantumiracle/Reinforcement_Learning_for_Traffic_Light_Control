
# Deep Q Network for traffic lights control
### (More about model description, see in _Intelligent Transportation System.md_)
[_Deep Reinforcement Learning for Intelligent Transportation Systems_ NIPS 2018 Workshop MLITS](https://openreview.net/forum?id=BJl846ey97)

The states transformation principle is shown in the graph:

<img src="https://github.com/quantumiracle/DQN_traffic_light_control/blob/master/states.png" width="25%">

__States__ representation:

0: Green 1 & Red 2\
1: Red 1 & Green 2\
2: Yellow 1 & Red 2\
3: Red 1 & Yellow 2

__Actions__ representation:

⓪: change state\
①: keep on

## one-way-two-queue intersection
It is one intersection with only two unidirectional roads, no left or right turning. The number of cars on each road is denoted as ![equ.1](https://latex.codecogs.com/gif.latex?Q_1,&space;Q_2\in&space;I) respectively. The state of the traffic light is denoted by state S, which can be in one of the following four states

* "0": green light for road Q_1, and hence red light for road Q_2;
* "1": red light for road Q_1, and hence green light for road Q_1;
* "2": yellow light for Q_1, and red light for road Q_2;
* "3": red light for road Q_1, and yellow light for road Q_2;

And the transition of states, which is called the action in RL algorithm, can be:

* "0": change;
* "1": keep on;

According to general transportation principles, the state transition of traffic lights could only be unidirectional, which is ![equ.1](https://latex.codecogs.com/gif.latex?"0"\rightarrow{"2"}\rightarrow{"1"}\rightarrow{"3"}\rightarrow{"0"}) under our definition of light states above. The trained RL agent takes the tuple [Q_1, Q_2, S] as input and generates action choice for traffic light.

<img src="https://github.com/quantumiracle/DQN_traffic_light_control/blob/master/one_way_two_queue.png" width="25%">
The training curve:
<img src="https://github.com/quantumiracle/DQN_traffic_light_control/blob/master/train.png" width="40%">

Code in **./one_way_two_queue/** with `<python light_constr.py --train>` to use.

## linear-network intersections
Noticing that we don't care much about the outcoming roads, which is denoted by dashed lines.
<img src="https://github.com/quantumiracle/DQN_traffic_light_control/blob/master/linear_network.png" width="40%">

States transformation in experiments: the color of lights is 'green' or 'red' or 'yellow'. The black rectangular represents incoming car for periphery of road networks. The numbers indicates number of cars on each road. If the light is 'green', the number of cars in that road will reduce the number of passing cars after transition. If there is 'black rectangular', the number of cars in the correspongding road will increase one after transition. The upper image is the state before transition, while the lower image is the state after transition. 

<img src="https://github.com/quantumiracle/DQN_traffic_light_control/blob/master/2*1.png" width="25%">


Code in **./linear_network/** with `<python lights.py --train>` to use.

## grid-square-network intersections
<img src="https://github.com/quantumiracle/DQN_traffic_light_control/blob/master/grid_square_network.png" width="40%">
States transformation in experiments:
<img src="https://github.com/quantumiracle/DQN_traffic_light_control/blob/master/4*4.png" width="50%">
The training curve:
<img src="https://github.com/quantumiracle/DQN_traffic_light_control/blob/master/2*2_100m.png" width="40%">

Code in **./one_way_two_queue/** with `<python lights.py --train>` to use.
