import numpy as np


import argparse
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pickle
import gzip
import time
import tkinter as tk
from env.cross import crossing
# from visual import Visual
import tensorflow as tf

np.set_printoptions(threshold=np.inf)

class Traffic_env:
    def __init__(
        self,
        grid_x,
        grid_y,
    
    ):
        self.grid_x=grid_x
        self.grid_y=grid_y
        self.grid_size=grid_x*grid_y

    def crossroads_map(self, x,y, q_states):
        cross={}  #dictionary of crossroads
        for xx in x:
            for yy in y:
                lab=str(xx)+str(yy)
                '''randomize intialization of light state!'''
                if np.random.rand()<0.25:
                    light=0
                elif np.random.rand()<0.5:
                    light=1
                elif np.random.rand()<0.75:
                    light=2
                else:
                    light=3
                '''initialize the car numbers on each road'''
                #1. random with upper bound 10:
                car_nums=[]
                for i in range(4):
                    car_nums.append(int(np.random.rand()*10))
                #2. fixed number 50: (for testing stabilization)
                # car_nums=[50,50,50,50]
                
                '''initialize the light state'''
                #1. random intialization of state
                cross_=crossing(car_nums=np.array(car_nums),light_state=light,q_states=q_states[xx][yy])
                #2. fixed initialization of 0 state
                # cross_=crossing(car_nums=np.array([0,0,0,0]),light_state=0,q_states=q_states[xx][yy])
                cross[lab]=cross_
        return cross  
        
    def env_init(self, ):
        ''' init the grid road network environment'''
        x=[]
        y=[]
        for i in range(self.grid_x):
            x.append(i+1)
        for i in range(self.grid_y):
            y.append(i+1)

        ## assign road type label with q_states
        q_states=[[([1] * 4) for i in range(self.grid_y+1)]for j in range(self.grid_x+1)]
        #print(q_states)
        for xx in x:
            for yy in y:
                #q_states is the inner/peripheral property of crossroads
                if xx==1:
                    q_states[xx][yy][2]=2 #0 for peripherial road, 1 for inner road, 2 for main road
                if xx==self.grid_x:
                    q_states[xx][yy][3]=2
                if yy==1:
                    q_states[xx][yy][0]=0
                if yy==self.grid_y:
                    q_states[xx][yy][1]=0
        
        cross=self.crossroads_map(x,y,q_states)
        return cross

    def reset(self,):
        ''' reset the road network environment'''
        x=[]
        y=[]
        for i in range(self.grid_x):
            x.append(i+1)
        for i in range(self.grid_y):
            y.append(i+1)
        obs=[]
        cross = self.env_init()
        for xx in x:
            for yy in y:
                lab=str(xx)+str(yy)
                obs=np.concatenate((obs, cross[lab].car_nums, cross[lab].light_state), axis=None)
        return np.array([obs]), cross


    def step(self,action, cross):
        ''' change the state of road network 'cross' according to the 'action' '''
        action_set=[[0 for i in range(self.grid_y+1)]for j in range(self.grid_x+1)]
        peri_cars=[[([0] * 4) for i in range(self.grid_y+1)]for j in range(self.grid_x+1)]
        in_cars=[[([0] * 4) for i in range(self.grid_y+1)]for j in range(self.grid_x+1)]
        #light state changes, cars numbers change, interactions between crossroads and peripherals
        x=[]
        y=[]
        for i in range(self.grid_x):
            x.append(i+1)
        for i in range(self.grid_y):
            y.append(i+1)

        # last_reward=0.
        # for xx in x:
        #     for yy in y:
        #         lab=str(xx)+str(yy)
        #         for i in range (4): 
        #             last_reward = last_reward - cross[lab].car_nums[i]**2


        for xx in x:
            for yy in y:
                lab=str(xx)+str(yy)
                '''no ecoding for action, directly output action for each intersection'''
                # action_set[xx][yy]=action[0][(xx-1)*self.grid_y+yy-1]
                action_set[xx][yy]=action[(xx-1)*self.grid_y+yy-1]
                '''
                pass_cars: total number of passing cars for road network during once state change;
                peri_cars: number of cars coming from each peripheral road of road network;
                in_cars: number of cars passing from last intersection to enter each intersection;
                '''
                pass_cars, peri_cars[xx][yy], in_cars[xx][yy]=cross[lab].state_change(action_set[xx][yy])
                # print('pass:', pass_cars)

    


        #interactions among crossroads        
        for xx in x:
            for yy in y:
                lab=str(xx)+str(yy)
                if cross[lab].q_states[0]==1:
                    if yy-1>0 and in_cars[xx][yy-1][0]>0:
                        cross_=cross[lab]
                        cross_.car_nums[0]+=in_cars[xx][yy-1][0]
                        cross[lab]=cross_
                        
                if cross[lab].q_states[1]==1:
                    if yy+1<=self.grid_y and  in_cars[xx][yy+1][1]>0:
                        cross_=cross[lab]
                        cross_.car_nums[1]+=in_cars[xx][yy+1][1]
                        cross[lab]=cross_
                       
                if cross[lab].q_states[2]==1:
                    if xx-1>0 and  in_cars[xx-1][yy][2]>0:
                        cross_=cross[lab]
                        cross_.car_nums[2]+=in_cars[xx-1][yy][2]
                        cross[lab]=cross_
                   
                if cross[lab].q_states[3]==1:
                    if xx+1<=self.grid_x and  in_cars[xx+1][yy][3]>0:
                        cross_=cross[lab]
                        cross_.car_nums[3]+=in_cars[xx+1][yy][3]
                        cross[lab]=cross_
        # cur_reward=0.
        # for xx in x:
        #     for yy in y:
        #         lab=str(xx)+str(yy)
        #         for i in range (4): 
        #             cur_reward = cur_reward - cross[lab].car_nums[i]**2

        reward=0.
        
        for xx in x:
            for yy in y:
                lab=str(xx)+str(yy)
                for i in range (4): 
                    reward = reward - cross[lab].car_nums[i]**2

        '''differential reward'''
        # reward=cur_reward-last_reward+pass_cars**2
        ''' 'pass_cars' reward'''
        # reward = pass_cars


        obs_=[]
        for xx in x:
            for yy in y:
                lab=str(xx)+str(yy)
                obs_=np.concatenate((obs_, cross[lab].car_nums, cross[lab].light_state), axis=None)
        return np.array([obs_]), np.array([reward]), cross, np.array([False])
    
    