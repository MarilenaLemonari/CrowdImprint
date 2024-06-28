# IMPORTS:
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from scipy.interpolate import interp1d

# INSTRUCTIONS:
#   cd C:\PROJECTS\SocialLandmarks
#   .venv/Scripts/activate
#   cd .\Data\
#   python3 .\data_recording.py

def load_npz(file_path):
    # file_path = 'C:\PROJECTS\SocialLandmarks\Data\PythonFiles\class_11_subject1.mp4_fish0.npz'
    data = np.load(file_path)
    # print("Keys in the .npz file:", data.files)
    # df = pd.DataFrame(data)
    x = data["X#wcentroid"]
    y = data["Y#wcentroid"]
    time = data["time"]
    x_adj = []
    y_adj = []
    time_adj = []
    c = 0
    for c in range(len(x)):
        i = x[c]
        j = y[c]
        t = time[c]
        if i != np.inf or j != np.inf:
            x_adj.append(i)
            y_adj.append(j)
            time_adj.append(t)
    return x_adj, y_adj, time_adj

source_x = 14.9505415
source_y = 5.143142
x, y, time = load_npz('C:\PROJECTS\SocialLandmarks\Data\PythonFiles\class_0_subject1.mp4_fish0.npz')
x2, y2, time2 = load_npz('C:\PROJECTS\SocialLandmarks\Data\PythonFiles\class_1_subject1.mp4_fish0.npz')
plt.plot(source_x, source_y, 'o')
plt.plot(x[0], y[0], "*")
plt.plot(x,y, '.')
plt.plot(x2[0], y2[0], "*")
plt.plot(x2,y2, '.')
plt.show()