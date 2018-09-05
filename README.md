# DQN_traffic_light_control

## Deep Q Network for raffic lights control
The states transformation principle is shown in the graph:

![States Transformation Graph](https://github.com/quantumiracle/DQN_traffic_light_control/blob/master/states.png)

States representation:

0: Green 1 & Red 2\
1: Red 1 & Green 2\
2: Yellow 1 & Red 2\
3: Red 1 & Yellow 2



Actions representation:

⓪: change state\
①: keep on

### one-way-two-queue intersection
![one-way-two-queue](https://github.com/quantumiracle/DQN_traffic_light_control/blob/master/one_way_two_queue.png)
### linear-network intersections
![linear-network](https://github.com/quantumiracle/DQN_traffic_light_control/blob/master/linear_network.png)
### grid-square-network intersections
![grid_square-network](https://github.com/quantumiracle/DQN_traffic_light_control/blob/master/grid_square_network.png)
