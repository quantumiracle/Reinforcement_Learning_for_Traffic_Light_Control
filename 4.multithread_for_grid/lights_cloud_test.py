import numpy as np

from RL_brain import DeepQNetwork
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pickle
import os
import gzip
import time
import tkinter as tk
from env import crossing
from visual import Visual
import threading as th
from threading import Thread
import urllib 
import tensorflow as tf
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Process
np.set_printoptions(threshold=np.inf)


parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)
args = parser.parse_args()


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
grid_x=4
grid_y=1

RL = DeepQNetwork(n_actions=2**(grid_x*grid_y),  
                  n_features=5*(grid_x*grid_y), 
                #   learning_rate=0.01, 
                  e_greedy=0.9,
                  replace_target_iter=100, memory_size=10000,
                  e_greedy_increment=0.001,)


window = tk.Tk()
window.title('my window')
window.geometry('1000x1000')
canvas = tk.Canvas(window, bg='white', height=1000, width=1000)

x=[]
y=[]
for i in range(grid_x):
    x.append(i+1)
for i in range(grid_y):
    y.append(i+1)

#parameters for visualizing
times=100
bias=6
bias_t=20
bias_=40
b=2
q_states=[[([1] * 4) for i in range(grid_y+1)]for j in range(grid_x+1)]

for xx in x:
    for yy in y:
        #q_states is the inner/peripheral property of crossroads
        if xx==1:
            q_states[xx][yy][2]=0 #0 for peripherial road, 1 for inner road
        if xx==grid_x:
            q_states[xx][yy][3]=0
        if yy==1:
            q_states[xx][yy][0]=0
        if yy==grid_y:
            q_states[xx][yy][1]=0
      

light_states=[[0 for i in range(grid_y+1)]for j in range(grid_x+1)]

def int2bin(n, count=24):  #10 -> binary
    """returns the binary of integer n, using count number of digits"""
    return "".join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def crossroads_map():
    cross={}  #dictionary of crossroads
    for xx in x:
        for yy in y:
            lab=str(xx)+str(yy)
            cross_=crossing(car_nums=np.array([0,0,0,0]),light_state=0,q_states=q_states[xx][yy])
            cross[lab]=cross_
    return cross  
            # cross[]={''}



def worker(train_steps):
    

    cross=crossroads_map()
    # visual=Visual()
    reward_set=[]
    obs=[]
    for xx in x:
        for yy in y:
            lab=str(xx)+str(yy)
            obs=np.concatenate((obs, cross[lab].car_nums, cross[lab].light_state), axis=None)

    for steps in range(train_steps):

        action=RL.choose_action(obs)
        action_set=[[0 for i in range(grid_y+1)]for j in range(grid_x+1)]
        peri_cars=[[([0] * 4) for i in range(grid_y+1)]for j in range(grid_x+1)]
        in_cars=[[([0] * 4) for i in range(grid_y+1)]for j in range(grid_x+1)]
        #light state changes, cars numbers change, interactions between crossroads and peripherals
        for xx in x:
            for yy in y:
                lab=str(xx)+str(yy)
                action_set[xx][yy]=int(int2bin(action,grid_x*grid_y)[(xx-1)*grid_y+yy-1])
                peri_cars[xx][yy], in_cars[xx][yy]=cross[lab].state_change(action_set[xx][yy])
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
        
        # visual.visual_peri(peri_cars,x,y,times,b,bias,bias_,bias_t,grid_x,grid_y)
        reward=0
        for xx in x:
            for yy in y:
                lab=str(xx)+str(yy)
                for i in range (4): 
                    reward = reward - cross[lab].car_nums[i]**2 
        # visual.visual_after(cross,x,y,times,b,bias,bias_t)

        obs_=[]
        for xx in x:
            for yy in y:
                lab=str(xx)+str(yy)
                obs_=np.concatenate((obs_, cross[lab].car_nums, cross[lab].light_state), axis=None)

        RL.store_transition(obs,action,reward,obs_)
        obs=obs_

def learn(lr_steps,learning_rate):
    for steps in range(lr_steps):
        RL.learn(learning_rate)

        



step_set=[]
reward_set=[]

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

if args.train:
    '''
    ###test### part of code is used to create a specific thread for 
    showing reward curve during training process
    '''
    train_steps=1000000
    lr_steps=1
    learning_rate=0.01
    workers = 3
    processes = []
    step_set=[]
    
    
    # multithreading--call #of workers for sampling (CPU)
    coord = tf.train.Coordinator()
    for p_number in range(workers):
        #using multi-thread instead of multi-process
        p=th.Thread(target=worker, args=(train_steps,))
        # p = Process(target=worker, args=(train_steps,))
        # p=ThreadWithReturnValue(target=worker, args=(train_steps,))
        p.daemon = True
        p.start()
        processes.append(p)

    
    
    ##########
    ###test###
    cross=crossroads_map()
    # visual=Visual()
    reward_set=[]
    obs=[]
    for xx in x:
        for yy in y:
            lab=str(xx)+str(yy)
            obs=np.concatenate((obs, cross[lab].car_nums, cross[lab].light_state), axis=None)
    ###test###
    ##########
    
    
    
    for step in range (train_steps):
        step_set.append(step)
        
        ##########
        ###test###
        action=RL.choose_action(obs)
        action_set=[[0 for i in range(grid_y+1)]for j in range(grid_x+1)]
        peri_cars=[[([0] * 4) for i in range(grid_y+1)]for j in range(grid_x+1)]
        in_cars=[[([0] * 4) for i in range(grid_y+1)]for j in range(grid_x+1)]
        #light state changes, cars numbers change, interactions between crossroads and peripherals
        for xx in x:
            for yy in y:
                lab=str(xx)+str(yy)
                action_set[xx][yy]=int(int2bin(action,grid_x*grid_y)[(xx-1)*grid_y+yy-1])
                peri_cars[xx][yy], in_cars[xx][yy]=cross[lab].state_change(action_set[xx][yy])

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
 
        
        # visual.visual_peri(peri_cars,x,y,times,b,bias,bias_,bias_t,grid_x,grid_y)
        reward=0
        for xx in x:
            for yy in y:
                lab=str(xx)+str(yy)
                for i in range (4): 
                    reward = reward - cross[lab].car_nums[i]**2

        obs_=[]
        for xx in x:
            for yy in y:
                lab=str(xx)+str(yy)
                obs_=np.concatenate((obs_, cross[lab].car_nums, cross[lab].light_state), axis=None)

        obs=obs_
        reward_set.append(reward)
        
        if step%20 ==0:
            print(step,',  ',reward)
            plt.plot(step_set,reward_set)
  
            plt.savefig('reward1.png')
        ###test###
        ##########


        #decay learning rate
        if step%40000==0:
            learning_rate=learning_rate*0.2
        if step%1000==0:
            RL.store()

        #call learning(GPU backpropagation) every 0.001 second during each step, 
        #therefore the training frequency is about 1000Hz
        time.sleep(0.001)
        q=th.Thread(target=learn, args=(lr_steps,learning_rate))
        # q=Process(target=learn, args=(lr_steps,learning_rate))
        # q=ThreadWithReturnValue(target=learn, args=(lr_steps,learning_rate))
        q.daemon = True
        q.start()

        processes.append(q)

    coord.join(processes)
    RL.store()

if args.test:
    RL.test_set()  # no exploration noise
    cross=crossroads_map()
    visual=Visual()
    obs=[]
    Q1_set=[]
    Q2_set=[]
    Q3_set=[]
    Q4_set=[]
    for xx in x:
        for yy in y:
            lab=str(xx)+str(yy)
            obs=np.concatenate((obs, cross[lab].car_nums, cross[lab].light_state), axis=None)
    RL.restore()
    for steps in range(1000):
        for xx in x:
            for yy in y:
                lab=str(xx)+str(yy)
        visual.visual_before(cross,x,y,times,b,bias,bias_t)

        action=RL.choose_action(obs)
        action_set=[[0 for i in range(grid_y+1)]for j in range(grid_x+1)]
        peri_cars=[[([0] * 4) for i in range(grid_y+1)]for j in range(grid_x+1)]
        in_cars=[[([0] * 4) for i in range(grid_y+1)]for j in range(grid_x+1)]
        #light state changes, cars numbers change, interactions between crossroads and peripherals
        for xx in x:
            for yy in y:
                lab=str(xx)+str(yy)
                action_set[xx][yy]=int(int2bin(action,grid_x*grid_y)[(xx-1)*grid_y+yy-1])
                peri_cars[xx][yy], in_cars[xx][yy]=cross[lab].state_change(action_set[xx][yy])
 
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

                if xx==1:
                    Q1_set.append(cross[lab].car_nums[2])
                elif xx==2:
                    Q2_set.append(cross[lab].car_nums[2])
                elif xx==3:
                    Q3_set.append(cross[lab].car_nums[2])
                elif xx==4:
                    Q4_set.append(cross[lab].car_nums[2])

        visual.visual_peri(peri_cars,x,y,times,b,bias,bias_,bias_t,grid_x,grid_y)

        reward=0
        
        for xx in x:
            for yy in y:
                lab=str(xx)+str(yy)
                for i in range (4): 
                    reward = reward - cross[lab].car_nums[i]**2

        visual.visual_after(cross,x,y,times,b,bias,bias_t)
        time.sleep(10)
        obs_=[]
        for xx in x:
            for yy in y:
                lab=str(xx)+str(yy)
                obs_=np.concatenate((obs_, cross[lab].car_nums, cross[lab].light_state), axis=None)
       


        if steps%50==0:
            print(steps,reward)
        if steps%100==0:
            plt.plot(step_set,reward_set)
            plt.savefig('test2.png')
        reward_set.append(reward)
        step_set.append(steps)

        obs=obs_
        
    window.mainloop()

    #plot for linear road network
    plot_len=100
    ax1=plt.subplot(2,1,1)
    plt.sca(ax1)
    plt.plot(step_set[:plot_len],reward_set[:plot_len])

    plt.ylabel('-Loss',fontsize=15)


    ax2=plt.subplot(2,1,2)
    plt.sca(ax2)


    plt.plot(step_set[:plot_len], Q1_set[:plot_len],'--',label='X1')
    plt.plot(step_set[:plot_len], Q2_set[:plot_len],label='X2')
    plt.plot(step_set[:plot_len], Q3_set[:plot_len],label='X3')
    plt.plot(step_set[:plot_len], Q4_set[:plot_len],label='X4')
    plt.xlabel('Steps',fontsize=15)
    plt.ylabel('Cars Numbers',fontsize=15)
    # set legend
    leg = plt.legend(loc=4,prop={'size': 15})
    legfm = leg.get_frame()
    legfm.set_edgecolor('black') # set legend fame color
    legfm.set_linewidth(0.5)   # set legend fame linewidth

    plt.savefig('test.png')

    plt.show()

        
        
