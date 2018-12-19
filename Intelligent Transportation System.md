# Intelligent Transportation System--Spotlights in Reinforcement Learning

# Objects choice:

-Intersection-based:

The intersections are defined to be the objects in you codes. And vehicles around it in a specific range are subjective property of the intersection. Our origin model choose this.

-Vehicle-based:

The vehicles are defined to be the objects in your codes. And intersections near the vehicles are kind of surrounding environments of them. As for in what range the vehicle needs to interact with a surrounding intersection, you can use spatial division of areas (kind of spatial distributed centers) with strict bundary as definition of area and pre-bundary (which can be overlapping among each other) as a perceptive area.

The object choice is depended on what important properties of the object you need to use in results display or statistics. Are they properties of the intersections or particularly each vehicle? Choose accordingly.

Above is also adaptive to similar model as centers with multiple units.

# Agent choice:

-Single agent

Use single agent as the overall control unit of the whole road network. Inputs of it is overall numerical information of the network. Therefore it has a global view, which means it has ability to reach the global optimal. However, the convergence of this model could be hard with a large state and action space.

-Multiple agents (without interactions)

Each agent for controlling a single intersection. They could be the same agent if all intersections are the same. And of course you can choose different agent for each intersection even they are the same. But no interactions means these agent could only have a local view of its surrounding roads, which usually means they could only reach a local optimal for the whole road network.

-Multiple agents (with interactions)

Interactions among multiple agents will definitely increase input information of each agent, which will broaden the view of each agent to reach a more global (more global still means local) optimal policy.

# Parallelism in training:

We could use multiple threads or multiple processes for accelerating the sampling process, if CPU consumption is too much compared with the GPU backpropagation process.（we did this in our multithread version model.） And what if GPU consumption is too much? As the network could only be updated in a single process (or thread), multiple processes updating the network will cause chaos. Should we copy the network for multiple processes backpropagation like we copy the environment in multiple processes sampling process?
# General in Reinforcement Learning:

-Why current reinforcement learning structure is not as efficient as human learning?

Notice that with limited computation ability of human brain, we human usually learn things or compute things in a relatively simple format. With attention, human computes problems simply through extraction and structuralization. In human's sight, the intersection information may be extracted to be two roads with some numbers of vehicles on them. While in machine's sight, it's just serial of numbers like '0102002030302...'. Therefore machine will definitely learn slower than human. A fair comparison should be that we eliminate the common knowledge part of the model, leave only the number serials for human to learn in the same way as machine does. Then compare the efficiency. 

That's to say, human actually take an __extraction and structuralization__ process unconsciously before learning  a specific model, which is actually based on __prior knowledge__. And this __extraction and structuralization__ process gives human the __attention__ scheme at the meantime.





For those knowledge you could search on Internet or directly through wikipedia, you don't need to learn or even remember at all. All that you need is to know there is something like that, and better have a fastest way to search and understand it. For this kind of knowledge, you put it in memory only if you will use it frequently in the near future. But if you use it frequently, you will naturally remember it. :)

