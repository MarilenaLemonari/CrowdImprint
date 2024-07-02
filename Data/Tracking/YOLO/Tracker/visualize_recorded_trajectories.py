import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import os
import csv

# python .\visualize_recorded_trajectories.py

traj_dir = './Trajectories/' 
traj_files = glob.glob(os.path.join(traj_dir, '*.csv'))
traj_files = ['./Trajectories/class_1_subject2.csv']

trajectories = []
plt.figure(figsize=(6, 6))
for traj_file in traj_files:
    traj_array = np.genfromtxt(traj_file, delimiter=',', skip_header=False)
    trajectories.append(traj_array)
    x = traj_array[:,1]
    y = traj_array[:,2]
    start_y = y[0]
    plt.plot(x,y, 'o-', markersize=3)
    plt.plot(traj_array[0,3], traj_array[1,3], '*', markersize = 5)

    # window_size = 10
    # smoothed_y = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
    # smoothed_x = x[(window_size//2):-(window_size//2)]
    # plt.plot(smoothed_x,smoothed_y, '-', markersize=3)

plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Feet Trajectories in World Space')
plt.grid(True)
plt.xlim([5, 17])
plt.ylim([0, 15])
plt.show()