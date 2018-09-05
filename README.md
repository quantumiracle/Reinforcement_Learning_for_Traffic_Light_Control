
# Deep Q Network for raffic lights control
The states transformation principle is shown in the graph:

<img src="https://github.com/quantumiracle/DQN_traffic_light_control/blob/master/states.png" width="25%">
**States** representation:

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
States transformation in experiments:
<img src="https://github.com/quantumiracle/DQN_traffic_light_control/blob/master/2*1.png" width="40%">


Code in **./linear_network/** with `<python lights.py --train>` to use.

## grid-square-network intersections
<img src="https://github.com/quantumiracle/DQN_traffic_light_control/blob/master/grid_square_network.png" width="40%">
States transformation in experiments:
<img src="https://github.com/quantumiracle/DQN_traffic_light_control/blob/master/4*4.png" width="40%">
The training curve:
<img src="https://github.com/quantumiracle/DQN_traffic_light_control/blob/master/2*2_100m.png" width="40%">

Code in **./one_way_two_queue/** with `<python lights.py --train>` to use.
