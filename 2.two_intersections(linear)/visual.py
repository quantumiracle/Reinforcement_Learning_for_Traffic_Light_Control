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
        window.geometry('500x500')
        self.canvas = tk.Canvas(window, bg='white', height=200, width=300)
        self.canvas_ = tk.Canvas(window, bg='white', height=200, width=300)

        
    
    def visual_before(self, cross1, cross2):
        # visualize
        if cross1.light_state==0:
            cross1_lr='red'
            cross1_ud='green'
        elif cross1.light_state==1:
            cross1_lr='green'
            cross1_ud='red'
        elif cross1.light_state==2:
            cross1_lr='red'
            cross1_ud='yellow'
        elif cross1.light_state==3:
            cross1_lr='yellow'
            cross1_ud='red'
        
        if cross2.light_state==0:
            cross2_lr='red'
            cross2_ud='green'
        elif cross2.light_state==1:
            cross2_lr='green'
            cross2_ud='red'
        elif cross2.light_state==2:
            cross2_lr='red'
            cross2_ud='yellow'
        elif cross2.light_state==3:
            cross2_lr='yellow'
            cross2_ud='red'

        self.canvas.delete('all')
        self.canvas.create_oval(75, 100, 80, 105, fill=cross1_lr) #left
        self.canvas.create_oval(125, 100, 130, 105, fill=cross1_lr) #right
        self.canvas.create_oval(100, 75, 105, 80, fill=cross1_ud) #up
        self.canvas.create_oval(100, 125, 105, 130, fill=cross1_ud) #down
        self.canvas.create_text(50, 100, text = cross1.car_nums[2],font=("bold", 22),fill = 'black')  #left
        self.canvas.create_text(100, 50, text = cross1.car_nums[0],font=("bold", 22),fill = 'black')  #up
        self.canvas.create_text(100, 150, text = cross1.car_nums[1],font=("bold", 22),fill = 'black') #down
        # self.canvas.create_text(150, 100, text = '1',fill = 'black')

        self.canvas.create_oval(175, 100, 180, 105, fill=cross2_lr) #left 
        self.canvas.create_oval(225, 100, 230, 105, fill=cross2_lr) #right
        self.canvas.create_oval(200, 75, 205, 80, fill=cross2_ud)  #up
        self.canvas.create_oval(200, 125, 205, 130, fill=cross2_ud) #down
        self.canvas.create_text(250, 100, text = cross2.car_nums[3],font=("bold", 22),fill = 'black')  #right
        self.canvas.create_text(200, 50, text = cross2.car_nums[0],font=("bold", 22),fill = 'black')   #up
        self.canvas.create_text(200, 150, text = cross2.car_nums[1],font=("bold", 22),fill = 'black')  #down
        self.canvas.create_text(150, 75, text = cross2.car_nums[2],font=("bold", 22),fill = 'black')   #left(up)
        self.canvas.create_text(150, 125, text = cross1.car_nums[3],font=("bold", 22),fill = 'black')  #left(down)

    def visual_peri(self, peri_cars1, peri_cars2):
        if peri_cars1[0]==1:
            self.canvas.create_rectangle(100, 30, 105,35, fill = 'black' )
        if peri_cars1[1]==1:
            self.canvas.create_rectangle(100, 165, 105,170, fill = 'black' )
        if peri_cars1[2]==1:
            self.canvas.create_rectangle(30, 100, 35,105, fill = 'black' )
        # if peri_cars1[3]==1:
        #     self.canvas_create_rectangle(155, 100, 160,105, fill = 'black' )

        if peri_cars2[0]==1:
            self.canvas.create_rectangle(200, 30, 205,35, fill = 'black' )
        if peri_cars2[1]==1:
            self.canvas.create_rectangle(200, 165, 205,170, fill = 'black' )
        # if peri_cars2[2]==1:
        #     self.canvas_create_rectangle(40, 100, 45,105, fill = 'black' )
        if peri_cars2[3]==1:
            self.canvas.create_rectangle(265, 100, 270,105, fill = 'black' )
        self.canvas.pack()
        self.canvas.update()

    def visual_after(self, cross1, cross2):
        if cross1.light_state==0:
            cross1_lr='red'
            cross1_ud='green'
        elif cross1.light_state==1:
            cross1_lr='green'
            cross1_ud='red'
        elif cross1.light_state==2:
            cross1_lr='red'
            cross1_ud='yellow'
        elif cross1.light_state==3:
            cross1_lr='yellow'
            cross1_ud='red'
        
        if cross2.light_state==0:
            cross2_lr='red'
            cross2_ud='green'
        elif cross2.light_state==1:
            cross2_lr='green'
            cross2_ud='red'
        elif cross2.light_state==2:
            cross2_lr='red'
            cross2_ud='yellow'
        elif cross2.light_state==3:
            cross2_lr='yellow'
            cross2_ud='red'

        self.canvas_.delete('all')
        self.canvas_.create_oval(75, 100, 80, 105, fill=cross1_lr) #left
        self.canvas_.create_oval(125, 100, 130, 105, fill=cross1_lr) #right
        self.canvas_.create_oval(100, 75, 105, 80, fill=cross1_ud) #up
        self.canvas_.create_oval(100, 125, 105, 130, fill=cross1_ud) #down
        self.canvas_.create_text(50, 100, text = cross1.car_nums[2],font=("bold", 22),fill = 'black')  #left
        self.canvas_.create_text(100, 50, text = cross1.car_nums[0],font=("bold", 22),fill = 'black')  #up
        self.canvas_.create_text(100, 150, text = cross1.car_nums[1],font=("bold", 22),fill = 'black') #down
        # self.canvas.create_text(150, 100, text = '1',fill = 'black')

        self.canvas_.create_oval(175, 100, 180, 105, fill=cross2_lr) #left 
        self.canvas_.create_oval(225, 100, 230, 105, fill=cross2_lr) #right
        self.canvas_.create_oval(200, 75, 205, 80, fill=cross2_ud)  #up
        self.canvas_.create_oval(200, 125, 205, 130, fill=cross2_ud) #down
        self.canvas_.create_text(250, 100, text = cross2.car_nums[3],font=("bold", 22),fill = 'black')  #right
        self.canvas_.create_text(200, 50, text = cross2.car_nums[0],font=("bold", 22),fill = 'black')   #up
        self.canvas_.create_text(200, 150, text = cross2.car_nums[1],font=("bold", 22),fill = 'black')  #down
        self.canvas_.create_text(150, 75, text = cross2.car_nums[2],font=("bold", 22),fill = 'black')   #left(up)
        self.canvas_.create_text(150, 125, text = cross1.car_nums[3],font=("bold", 22),fill = 'black')  #left(down)
        self.canvas_.pack()
        self.canvas_.update()
