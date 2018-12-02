# **one_agent_for_each_intersection**

- ./env/cross.py: the class of single intersection with four bidirectional roads, as the algorithm is intersection based instead of vehicle based. 
- ./env/traffic_env.py: the class of overall traffic light environment for x rows and y columns grid road network .
- run.py: apply DDPG agent to learn and control, the common one agent for each intersection, instead of one agent for the overall networks; with args: `test` and `train`.
- other files: define the DDPG agent as in Openai baselines.
- `python -m run.py --alg=ddpg --num_timesteps=1e4 --train/retrain/test` to train, retrain or test.
