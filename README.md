
# Deep Q Network for raffic lights control
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
<img src="https://github.com/quantumiracle/DQN_traffic_light_control/blob/master/train.png" width="25%">
Code in ```./one_way_two_queue/```

## linear-network intersections
Noticing that we don't care much about the outcoming roads, which is denoted by dashed lines.
<img src="https://github.com/quantumiracle/DQN_traffic_light_control/blob/master/linear_network.png" width="40%">
Code in ```./one_way_two_queue/```

## grid-square-network intersections
<img src="https://github.com/quantumiracle/DQN_traffic_light_control/blob/master/grid_square_network.png" width="40%">
Code in ```./one_way_two_queue/```
