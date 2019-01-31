import numpy as np

from RL_brain2 import DeepQNetwork
#from cnn_brain import DeepQNetwork
import argparse
import matplotlib.pyplot as plt
import pickle
import gzip
import time
import tkinter as tk
from env import crossing
from visual import Visual
np.set_printoptions(threshold=np.inf)

#print(env.observation_space.shape[0])
parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)
args = parser.parse_args()








def road_map():

    cross1=crossing(light_state=0,q_states=[0,0,0,1])
    cross2=crossing(light_state=0,q_states=[0,0,1,0])


step_set=[]
reward_set=[]



if args.train:

    RL = DeepQNetwork(n_actions=4,  #2*2
                    #n_features=env.observation_space.shape[0],
                    n_features=10, #2*5
                    learning_rate=0.01, e_greedy=0.9,
                    replace_target_iter=100, memory_size=2000,
                    e_greedy_increment=0.001,)
    
    # hit control:
    # on_hit = False
    # next_step=0
    # def hit_me():
    #     global on_hit
    #     global next_step
    #     if on_hit == False:
    #         on_hit = True
    #         next_step=0
            
    #     else:
    #         on_hit = False
    #         next_step=1
    # b = tk.Button(window, text='Next Step', width=15,
    #             height=2, command=hit_me)
    # b.pack()
    visual=Visual()

    cross1=crossing(light_state=0,q_states=[0,0,0,1])
    cross2=crossing(light_state=0,q_states=[0,0,1,0])
    obs=np.concatenate((cross1.car_nums, cross1.light_state, cross2.car_nums, cross2.light_state),axis=None)
    for steps in range(20000):
        
        visual.visual_before(cross1, cross2)

        action=RL.choose_action(obs)
        if action == 0:
            action1 =0
            action2 =0
        elif action ==1:
            action1 =1
            action2 =0
        elif action ==2:
            action1 =0
            action2 =1
        elif action ==3:
            action1 =1
            action2 =1

        #interaction between crossroads and interaction between crossroad and peripheral
        peri_cars1, in_cars1 = cross1.state_change(action1)

        peri_cars2, in_cars2 = cross2.state_change(action2)
        # print(peri_cars1, in_cars1, peri_cars2, in_cars2)
        
        visual.visual_peri(peri_cars1,peri_cars2)

        reward=0
        for i in range (4):
            if cross2.q_states[i]==1:
                cross2.car_nums[i]+=in_cars1
            if cross1.q_states[i]==1:
                cross1.car_nums[i]+=in_cars2
            reward = reward - cross1.car_nums[i]**2 - cross2.car_nums[i]**2
        
        visual.visual_after(cross1, cross2)

        obs_=np.concatenate((cross1.car_nums, cross1.light_state, cross2.car_nums, cross2.light_state),axis=None)
        RL.store_transition(obs,action,reward,obs_)
        if steps>200:
            RL.learn()
        if steps%50==0:
            print(steps,reward)
            
        reward_set.append(reward)
        step_set.append(steps)
        #plt.scatter(steps, reward)
        obs=obs_
        
    # window.mainloop()
    plt.plot(step_set,reward_set)
    plt.savefig('train2.png')
    RL.store()
    plt.show()
    #RL.plot_cost()
if args.test:
    RL = DeepQNetwork(n_actions=4,  #2*2
                    #n_features=env.observation_space.shape[0],
                    n_features=10, #2*5
                    learning_rate=0.01, e_greedy=1.,
                    replace_target_iter=100, memory_size=2000,
                    e_greedy_increment=None,)
    cross1=crossing(light_state=0,q_states=[0,0,0,1])
    cross2=crossing(light_state=0,q_states=[0,0,1,0])
    RL.restore()
    for steps in range(1000):
        obs=np.concatenate((cross1.car_nums, cross1.light_state, cross2.car_nums, cross2.light_state),axis=None)
        # print(obs)
        action=RL.choose_action(obs)
        if action == 0:
            action1 =0
            action2 =0
        elif action ==1:
            action1 =1
            action2 =0
        elif action ==2:
            action1 =0
            action2 =1
        elif action ==3:
            action1 =1
            action2 =1

        peri_cars1, in_cars1 = cross1.state_change(action1)
        peri_cars2, in_cars2 = cross2.state_change(action2)
        reward=0
        for i in range (4):
            if cross2.q_states[i]==1:
                cross2.car_nums[i]+=in_cars1
            if cross1.q_states[i]==1:
                cross1.car_nums[i]+=in_cars2
            reward = reward - cross1.car_nums[i]**2 - cross2.car_nums[i]**2

        obs_=np.concatenate((cross1.car_nums, cross1.light_state, cross2.car_nums, cross2.light_state),axis=None)

        if steps%50==0:
            print(reward)
        
        obs=obs_
        steps+=1
        reward_set.append(reward)
        step_set.append(steps)
    plt.plot(step_set,reward_set)
    plt.savefig('test2.png')
    plt.show()
    

        
        
