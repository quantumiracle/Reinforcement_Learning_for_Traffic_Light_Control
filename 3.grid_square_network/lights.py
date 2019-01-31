import numpy as np

from RL_brain2 import DeepQNetwork

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
parser.add_argument('--train', dest='train', action='store_true', default=True)
# parser.add_argument('--test', dest='test', action='store_true', default=True)
args = parser.parse_args()


# size of crossroads grid
grid_x=4
grid_y=4

RL = DeepQNetwork(n_actions=2**(grid_x*grid_y),  #0,1 for each crossroad
                  n_features=5*(grid_x*grid_y), #2*5 (5 = 4 numbers of cars + 1 light state)
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001,)


x=[]
y=[]
for i in range(grid_x):
    x.append(i+1)
for i in range(grid_y):
    y.append(i+1)

#property of visualization
times=100 #interval: crossroad & crossroad
bias=6    #distance: light & center of crossroad
bias_t=20 #distance: text(car_number) & center of crossroad
bias_=40  #distance: coming car & center of crossroad
b=2       #size of oval & rectangle 

#q_states is the inner/peripheral property of crossroads
q_states=[[([1] * 4) for i in range(grid_y+1)]for j in range(grid_x+1)]
for xx in x:
    for yy in y:
        if xx==1:
            q_states[xx][yy][2]=0 #0 for peripherial road, 1 for inner road
        if xx==grid_x:
            q_states[xx][yy][3]=0
        if yy==1:
            q_states[xx][yy][0]=0
        if yy==grid_y:
            q_states[xx][yy][1]=0
        
#initial light_states of crossroads
light_states=[[0 for i in range(grid_y+1)]for j in range(grid_x+1)]

def int2bin(n, count=24):  #10 -> binary
    """returns the binary of integer n, using count number of digits"""
    return "".join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def crossroads_map(x,y):
    cross={}  #dictionary of crossroads
    for xx in x:
        for yy in y:
            lab=str(xx)+str(yy)  #string lab for addressing each crossroad, xx ~ [1,grid_x]
            #Initialization
            cross_=crossing(car_nums=np.array([0,0,0,0]),light_state=0,q_states=q_states[xx][yy])
            cross[lab]=cross_
    return cross  


step_set=[]
reward_set=[]

if args.train:
    #Initializing 
    cross=crossroads_map(x,y)
    visual=Visual()
    obs=[]
    for xx in x:
        for yy in y:
            lab=str(xx)+str(yy)
            obs=np.concatenate((obs, cross[lab].car_nums, cross[lab].light_state), axis=None)
    
    #Training steps
    for steps in range(200000):

        visual.visual_before(cross,x,y,times,b,bias,bias_t)

        action=RL.choose_action(obs)
        action_set=[[0 for i in range(grid_y+1)]for j in range(grid_x+1)]
        peri_cars=[[([0] * 4) for i in range(grid_y+1)]for j in range(grid_x+1)]
        in_cars=[[([0] * 4) for i in range(grid_y+1)]for j in range(grid_x+1)]

        #light state changes, cars numbers change, interactions between crossroads and peripherals
        for xx in x:
            for yy in y:
                lab=str(xx)+str(yy)
                #10->binary coding for action(1 value), like if action=128, 9 bits binary coding
                #of it is 010000000, indicating a 3*3 grid with each crossroad having action of 
                #'0''1''0''0''0''0''0''0''0'(storing in action set), each action is either '0' or '1',
                #for 'change state' or 'keep on'. The binary number is set to be (grid_x*grid_y) bits, 
                # correspongding to the number of crossroads in grid. 
                action_set[xx][yy]=int(int2bin(action,grid_x*grid_y)[(xx-1)*grid_y+yy-1])
                peri_cars[xx][yy], in_cars[xx][yy]=cross[lab].state_change(action_set[xx][yy])
                
        #interactions among crossroads        
        for xx in x:
            for yy in y:
                lab=str(xx)+str(yy)
                if cross[lab].q_states[0]==1:
                    if yy-1>0 and in_cars[xx][yy-1][0]>0: #having incoming cars from neighbors
                        cross_=cross[lab]
                        cross_.car_nums[0]+=in_cars[xx][yy-1][0]
                        cross[lab]=cross_

                if cross[lab].q_states[1]==1:
                    if yy+1<=grid_y and  in_cars[xx][yy+1][1]>0:
                        cross_=cross[lab]
                        cross_.car_nums[1]+=in_cars[xx][yy+1][1]
                        cross[lab]=cross_

                if cross[lab].q_states[2]==1:
                    if xx-1>0 and  in_cars[xx-1][yy][2]>0:
                        cross_=cross[lab]
                        cross_.car_nums[2]+=in_cars[xx-1][yy][2]
                        cross[lab]=cross_

                if cross[lab].q_states[3]==1:
                    if xx+1<=grid_x and  in_cars[xx+1][yy][3]>0:
                        cross_=cross[lab]
                        cross_.car_nums[3]+=in_cars[xx+1][yy][3]
                        cross[lab]=cross_
        #in the same diagram as the above 'visual_before', showing the incoming cars
        visual.visual_peri(peri_cars,x,y,times,b,bias,bias_,bias_t,grid_x,grid_y)

        reward=0
        for xx in x:
            for yy in y:
                lab=str(xx)+str(yy)
                for i in range (4): 
                    reward = reward - cross[lab].car_nums[i]**2
        #show the result of state tranformation in another diagram 'visual_after', i.e. the result of 'visual_before'
        visual.visual_after(cross,x,y,times,b,bias,bias_t)
        time.sleep(10)

        obs_=[]
        for xx in x:
            for yy in y:
                lab=str(xx)+str(yy)
                obs_=np.concatenate((obs_, cross[lab].car_nums, cross[lab].light_state), axis=None)
       
        RL.store_transition(obs,action,reward,obs_)
        
        if steps>200:
            RL.learn()
        if steps%50==0:
            print(steps,reward)
            
        reward_set.append(reward)
        step_set.append(steps)
        #plt.scatter(steps, reward)
	obs=obs_
        

    plt.plot(step_set,reward_set)
    plt.savefig('train2.png')
    RL.store()
    plt.show()
    #RL.plot_cost()

    

    

        
        
