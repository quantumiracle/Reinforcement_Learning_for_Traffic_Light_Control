
# Deep Q Network for traffic lights control
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
