import numpy as np
import argparse
import matplotlib.pyplot as plt
import pickle
import gzip
import time
import tkinter as tk

class Visual:
    def __init__(
            self,
    ):
        window = tk.Tk()
        window.title('my window')
        window.geometry('1000x1000')
        self.canvas = tk.Canvas(window, bg='white', height=200, width=500)

        self.canvas_ = tk.Canvas(window, bg='white', height=200, width=500)

        
    
    def visual_before(self, cross,x,y,times,b,bias,bias_t):
        self.canvas.delete('all')
        for xx in x:
            for yy in y:
                lab=str(xx)+str(yy)
                if cross[lab].light_state==0:
                    cross_lr='red'
                    cross_ud='green'
                elif cross[lab].light_state==1:
                    cross_lr='green'
                    cross_ud='red'
                elif cross[lab].light_state==2:
                    cross_lr='red'
                    cross_ud='yellow'
                elif cross[lab].light_state==3:
                    cross_lr='yellow'
                    cross_ud='red'
                # print(lab,cross[lab].car_nums )
                self.canvas.create_oval(times*xx-b-bias, times*yy-b, times*xx+b-bias, times*yy+b, fill=cross_lr)
                self.canvas.create_oval(times*xx-b+bias, times*yy-b, times*xx+b+bias, times*yy+b, fill= cross_lr)
                self.canvas.create_oval(times*xx-b, times*yy-b-bias, times*xx+b, times*yy+b-bias, fill=cross_ud)
                self.canvas.create_oval(times*xx-b, times*yy-b+bias, times*xx+b, times*yy+b+bias, fill=cross_ud)

                self.canvas.create_text(times*xx-bias_t, times*yy,text=cross[lab].car_nums[2])
                self.canvas.create_text(times*xx+bias_t, times*yy,text=cross[lab].car_nums[3])
                self.canvas.create_text(times*xx, times*yy-bias_t,text=cross[lab].car_nums[0])
                self.canvas.create_text(times*xx, times*yy+bias_t,text=cross[lab].car_nums[1])
        

    def visual_peri(self, peri_cars,x,y,times,b,bias,bias_,bias_t,grid_x,grid_y):
        for xx in x:
            for yy in y:
                if xx==1 and peri_cars[xx][yy][2]>0:
                    self.canvas.create_rectangle(times*xx-b-bias_, times*yy-b, times*xx+b-bias_, times*yy+b,fill = 'black')
                if xx==grid_x and peri_cars[xx][yy][3]>0:
                    self.canvas.create_rectangle(times*xx-b+bias_, times*yy-b, times*xx+b+bias_, times*yy+b,fill = 'black')
                if yy==1 and peri_cars[xx][yy][0]>0:
                    self.canvas.create_rectangle(times*xx-b, times*yy-b-bias_, times*xx+b, times*yy+b-bias_,fill = 'black')
                if yy==grid_y and peri_cars[xx][yy][1]>0:
                    self.canvas.create_rectangle(times*xx-b, times*yy-b+bias_, times*xx+b, times*yy+b+bias_,fill = 'black')
                # if xx == 1:
                #     if yy==1:
                #         if peri_cars[xx][yy][0]>0:
                #             self.canvas.create_rectangle(times*xx-b, times*yy-b-bias_, times*xx+b, times*yy+b-bias_, fill = 'black')
                #         if peri_cars[xx][yy][2]>0:
                #             self.canvas.create_rectangle(times*xx-b-bias_, times*yy-b, times*xx+b-bias_, times*yy+b,fill = 'black')
                    
                #     else:
                #         if peri_cars[xx][yy][2]>0:
                #             self.canvas.create_rectangle(times*xx-b-bias_, times*yy-b, times*xx+b-bias_, times*yy+b,fill = 'black')
                        
                # elif xx == grid_x:
                #     if yy==grid_y:
                #         if peri_cars[xx][yy][1]>0:
                #             self.canvas.create_rectangle(times*xx-b, times*yy-b+bias_, times*xx+b, times*yy+b+bias_, fill = 'black')
                #         if peri_cars[xx][yy][3]>0:
                #             self.canvas.create_rectangle(times*xx-b+bias_, times*yy-b, times*xx+b+bias_, times*yy+b,fill = 'black')
                #     else: 
                #         if peri_cars[xx][yy][3]>0:
                #             self.canvas.create_rectangle(times*xx-b+bias_, times*yy-b, times*xx+b+bias_, times*yy+b,fill = 'black')

                # if yy==1:
                #     if xx!=1:
                #         if peri_cars[xx][yy][0]>0:
                #             self.canvas.create_rectangle(times*xx-b, times*yy-b-bias_, times*xx+b, times*yy+b-bias_,fill = 'black')

                # if yy==grid_y:
                #     if  xx!=grid_x:
                #         if peri_cars[xx][yy][1]>0:
                #             self.canvas.create_rectangle(times*xx-b, times*yy-b+bias_, times*xx+b, times*yy+b+bias_,fill = 'black')
    
        self.canvas.pack()
        self.canvas.update()

    def visual_after(self, cross,x,y,times,b,bias,bias_t):
        self.canvas_.delete('all')
        for xx in x:
            for yy in y:
                lab=str(xx)+str(yy)
                if cross[lab].light_state==0:
                    cross_lr='red'
                    cross_ud='green'
                elif cross[lab].light_state==1:
                    cross_lr='green'
                    cross_ud='red'
                elif cross[lab].light_state==2:
                    cross_lr='red'
                    cross_ud='yellow'
                elif cross[lab].light_state==3:
                    cross_lr='yellow'
                    cross_ud='red'
        
                self.canvas_.create_oval(times*xx-b-bias, times*yy-b, times*xx+b-bias, times*yy+b, fill=cross_lr)
                self.canvas_.create_oval(times*xx-b+bias, times*yy-b, times*xx+b+bias, times*yy+b, fill= cross_lr)
                self.canvas_.create_oval(times*xx-b, times*yy-b-bias, times*xx+b, times*yy+b-bias, fill=cross_ud)
                self.canvas_.create_oval(times*xx-b, times*yy-b+bias, times*xx+b, times*yy+b+bias, fill=cross_ud)

                self.canvas_.create_text(times*xx-bias_t, times*yy,text=cross[lab].car_nums[2])
                self.canvas_.create_text(times*xx+bias_t, times*yy,text=cross[lab].car_nums[3])
                self.canvas_.create_text(times*xx, times*yy-bias_t,text=cross[lab].car_nums[0])
                self.canvas_.create_text(times*xx, times*yy+bias_t,text=cross[lab].car_nums[1])
        self.canvas_.pack()
        self.canvas_.update()
