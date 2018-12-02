# **one_agent_for_each_intersection**

- env.py: define the traffic light environment for x rows and y columns grid road network, each intersection with four bidirectional roads.
- lights.py: apply DQN agent to learn and control, the common one agent for each intersection, instead of one agent for the overall networks; with args: `test` and `train`.
- RL_brain.py: class of DQN algorithms with 4-layer neural network as eval net and target net.
- visual.py: class of functions for visualizing the dynamic network situations.
- `python lights.py --train/test` to train and test,  ``python lights_re.py --train`` to retrain after the pretraining. 
