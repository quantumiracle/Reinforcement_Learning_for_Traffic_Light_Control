# **multithread_program_for_grid_network**

- env.py: define the traffic light environment for x rows and y columns grid road network, each intersection with four bidirectional roads.
- lights_cloud_test.py: apply DQN agent to learn and control, with args: `test` to test results trained on the server with multithread acceleration algorithms.

  RL_brain.py: class of DQN algorithms with 4-layer neural network as eval net and target net.
- visual.py: class of functions for visualizing the dynamic network situations.
- `python lights_cloud_test.py --train` to train and ``python lights_cloud_test.py --test`` to test. 
