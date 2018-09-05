import numpy as np
import argparse
import matplotlib.pyplot as plt
import pickle
import gzip
import time
import tkinter as tk

class crossing:
    def __init__(
            self,
            light_state,
            q_states,
            car_nums=np.array([0,0,0,0]),
    ):
        self.car_nums=car_nums
        self.light_state=light_state
        self.q_states=q_states
        self.num_pass=2
        self.coming_prob=0.25
    def state_change(self, action):
        peri_cars=[]
        #jam detection
        for car_num in self.car_nums:
            if car_num > 10:
                print('Jam! Clear')
                self.car_nums =np.array([0,0,0,0])
                break

        #car in (peripherial road)
        i=0
        for q_state in self.q_states:
            if q_state==0:   #0 for peripherial road, 1 for inner road
                if np.random.random()<self.coming_prob:
                    self.car_nums[i]+=1
                    peri_cars.append(1)
                else:
                    peri_cars.append(0)
            else: peri_cars.append(0)
            i+=1    
        
        #light takes action
        if action ==0:  #change
            if self.light_state==0:
                self.light_state=2
                if self.car_nums[0]>(self.num_pass-1):
                    self.car_nums[0]-=self.num_pass
                    if self.q_states[1]==1:
                        ret0 = 1
                    else:
                        ret0 = 0
                else:
                    ret0 = 0

                if self.car_nums[1]>(self.num_pass-1):
                    self.car_nums[1]-=self.num_pass
                    if self.q_states[0]==1:
                        ret1 = 1
                    else:
                        ret1 = 0
                else:
                    ret1 = 0

                if ret0==1 or ret1==1:
                    ret = 1
                else: ret = 0
                

            elif self.light_state==1:
                self.light_state=3
                if self.car_nums[2]>(self.num_pass-1):
                    self.car_nums[2]-=self.num_pass
                    if self.q_states[3]==1:
                        ret2 = 1
                    else: 
                        ret2 = 0
                else:
                    ret2 = 0

                if self.car_nums[3]>(self.num_pass-1):
                    
                    self.car_nums[3]-=self.num_pass
                    if self.q_states[2]==1:
                        ret3 = 1
                    else:
                        ret3 = 0
                else:
                    ret3 = 0
                
                if ret2==1 or ret3==1:
                    ret = 1
                else: ret = 0
                

            elif self.light_state==2:
                self.light_state=1
                ret = 0

            elif self.light_state==3:
                self.light_state=0
                ret = 0

        elif action == 1:  #keep on
            if self.light_state==0:
                if self.car_nums[0]>(self.num_pass-1):
                    self.car_nums[0]-=self.num_pass
                    if self.q_states[1]==1:
                        ret0 = 1
                    else:
                        ret0 = 0
                else:
                    ret0 = 0

                if self.car_nums[1]>(self.num_pass-1):
                    self.car_nums[1]-=self.num_pass
                    if self.q_states[0]==1:
                        ret1 = 1
                    else:
                        ret1 = 0
                else:
                    ret1 = 0
                
                if ret0==1 or ret1==1:
                    ret = 1
                else: ret = 0

            elif self.light_state==1:
                # print('state1', self.car_nums[3])
                if self.car_nums[2]>(self.num_pass-1):
                    self.car_nums[2]-=self.num_pass
                    if self.q_states[3]==1:
                        ret2 = 1
                    else:
                        ret2 = 0
                else:
                    ret2 = 0

                if self.car_nums[3]>(self.num_pass-1):
                    # print('-----')
                    self.car_nums[3]-=self.num_pass
                    if self.q_states[2]==1:
                        ret3 = 1
                    else:
                        ret3 = 0
                else:
                    ret3 = 0

                if ret2==1 or ret3==1:
                    ret = 1
                else: ret = 0

            else:
                ret = 0

        else:
            print('Action error!')
            ret = 0
        return peri_cars, ret


    def visual_init(self):
        window = tk.Tk()
        window.title('my window')
        window.geometry('500x500')
        self.canvas = tk.Canvas(window, bg='white', height=200, width=300)
        self.canvas_ = tk.Canvas(window, bg='white', height=200, width=300)

