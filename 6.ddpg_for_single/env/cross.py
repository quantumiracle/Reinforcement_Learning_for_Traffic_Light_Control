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
            car_nums,
    ):
        self.car_nums=car_nums
        self.light_state=light_state
        self.q_states=q_states
        '''for ddpg to converge: num_pass should be almost twice as num_coming to avoid car accumulating'''
        pass_com_ratio=2  # passing rate / coming rate
        # self.num_pass=16  # num of passing cars for single green light
        # self.num_pass_branch=4
        self.coming_car_branch_upper=2.  #upper bound number of coming cars for single step single branch road
        self.coming_car_main_upper=8.  #upper bound number of coming cars for single step single main road
        self.num_pass=self.coming_car_main_upper*pass_com_ratio
        self.num_pass_branch=self.coming_car_branch_upper*pass_com_ratio

        # self.coming_prob=0.25 # cars coming rate from peripheral roads
    def state_change(self, action):
        pass_cars=0
        #jam detection
        for car_num in self.car_nums:
            if car_num > 100:
                # print('Jam! Clear')
                self.car_nums =np.array([0,0,0,0])
                break

        #car in (peripherial road)
        i=0
        peri_cars=[]
        for q_state in self.q_states:
            if q_state==0:   #0 for peripherial road, 1 for inner road
                # if np.random.random()<self.coming_prob:
                #     # print('comming car from ', i)
                #     self.car_nums[i]+=1
                #     peri_cars.append(1)
                # else:
                #     peri_cars.append(0)
                coming_cars=int(np.random.random()*self.coming_car_branch_upper)
                self.car_nums[i]+=coming_cars
                peri_cars.append(coming_cars)
            elif q_state==2:  # 2 for main road
                coming_cars=int(np.random.random()*self.coming_car_main_upper)
                self.car_nums[i]+=coming_cars
                peri_cars.append(coming_cars)
            else: peri_cars.append(0)
            i+=1    
        
        #passing cars for inner road
        ret0=0
        ret1=0
        ret2=0
        ret3=0

        if action ==0:  #change car number according to current light state, not next state
            if self.light_state==0:
                self.light_state=2
                if self.car_nums[0] > self.num_pass_branch:
                    self.car_nums[0]-=self.num_pass_branch
                    pass_cars+=self.num_pass_branch
                    if self.q_states[1]==1:  # if the contradict side is inner, record it (using ret_)
                        ret0 = self.num_pass_branch
                    else:
                        ret0 = 0
                else:
                    pass_cars+=self.car_nums[0]
                    if self.q_states[1]==1:
                        ret0 = self.car_nums[0]
                    else:
                        ret0 = 0
                    self.car_nums[0]=0

                if self.car_nums[1]>(self.num_pass_branch):
                    self.car_nums[1]-=self.num_pass_branch
                    pass_cars+=self.num_pass_branch
                    if self.q_states[0]==1:
                        ret1 = self.num_pass_branch
                    else:
                        ret1 = 0
                else:
                    pass_cars+=self.car_nums[1]
                    if self.q_states[0]==1:
                        ret1 = self.car_nums[1]
                    else:
                        ret1 = 0
                    self.car_nums[1]=0


                

            elif self.light_state==1:
                self.light_state=3
                if self.car_nums[2]>(self.num_pass):
                    self.car_nums[2]-=self.num_pass
                    pass_cars+=self.num_pass
                    if self.q_states[3]==1:
                        ret2 = self.num_pass
                    else: 
                        ret2 = 0
                else:
                    pass_cars+=self.car_nums[2]
                    if self.q_states[3]==1:
                        ret2 = self.car_nums[2]
                    else:
                        ret2 = 0
                    self.car_nums[2]=0


                if self.car_nums[3]>(self.num_pass):
                    
                    self.car_nums[3]-=self.num_pass
                    pass_cars+=self.num_pass
                    if self.q_states[2]==1:
                        ret3 = self.num_pass
                    else:
                        ret3 = 0
                else:
                    pass_cars+=self.car_nums[3]
                    if self.q_states[2]==1:
                        ret3 = self.car_nums[3]
                    else:
                        ret3 = 0
                    self.car_nums[3]=0
                
                # if ret2==1 or ret3==1:
                #     ret = 1
                # else: ret = 0
                

            elif self.light_state==2:
                self.light_state=1
                ret0=0
                ret1=0
                ret2=0
                ret3=0

            elif self.light_state==3:
                self.light_state=0
                ret0=0
                ret1=0
                ret2=0
                ret3=0
            else:
                ret0=0
                ret1=0
                ret2=0
                ret3=0

        elif action == 1:  #keep on
            if self.light_state==0:
                if self.car_nums[0]>(self.num_pass_branch-1):
                    self.car_nums[0]-=self.num_pass_branch
                    pass_cars+=self.num_pass_branch
                    if self.q_states[1]==1:
                        ret0 = self.num_pass_branch
                    else:
                        ret0 = 0
                else:
                    pass_cars+=self.car_nums[0]
                    if self.q_states[1]==1:
                        ret0 = self.car_nums[0]
                    else:
                        ret0 = 0
                    self.car_nums[0]=0

                if self.car_nums[1]>(self.num_pass_branch):
                    self.car_nums[1]-=self.num_pass_branch
                    pass_cars+=self.num_pass_branch
                    if self.q_states[0]==1:
                        ret1 = self.num_pass_branch
                    else:
                        ret1 = 0
                else:
                    pass_cars+=self.car_nums[1]
                    if self.q_states[0]==1:
                        ret1 = self.car_nums[1]
                    else:
                        ret1 = 0
                    self.car_nums[1]=0
                
                # if ret0==1 or ret1==1:
                #     ret = 1
                # else: ret = 0

            elif self.light_state==1:
                # print('state1', self.car_nums[3])
                if self.car_nums[2]>(self.num_pass-1):
                    self.car_nums[2]-=self.num_pass
                    pass_cars+=1
                    if self.q_states[3]==1:
                        ret2 = self.num_pass
                    else:
                        ret2 = 0
                else:
                    pass_cars+=self.car_nums[2]
                    if self.q_states[3]==1:
                        ret2 = self.car_nums[2]
                    else:
                        ret2 = 0
                    self.car_nums[2]=0

                if self.car_nums[3]>(self.num_pass-1):
                    # print('-----')
                    self.car_nums[3]-=self.num_pass
                    pass_cars+=1
                    if self.q_states[2]==1:
                        ret3 = self.num_pass
                    else:
                        ret3 = 0
                else:
                    pass_cars+=self.car_nums[3]
                    if self.q_states[2]==1:
                        ret3 = self.car_nums[3]
                    else:
                        ret3 = 0
                    self.car_nums[3]=0

                # if ret2==1 or ret3==1:
                #     ret = 1
                # else: ret = 0

            else:
                ret0=0
                ret1=0
                ret2=0
                ret3=0

        else:
            # print('Action error!')
            ret0=0
            ret1=0
            ret2=0
            ret3=0
        return pass_cars, peri_cars, [ret0,ret1,ret2,ret3]

    #visualize the simulation during experiments
    def visual_init(self):
        window = tk.Tk()
        window.title('my window')
        window.geometry('500x500')
        self.canvas = tk.Canvas(window, bg='white', height=200, width=300)
        self.canvas_ = tk.Canvas(window, bg='white', height=200, width=300)

