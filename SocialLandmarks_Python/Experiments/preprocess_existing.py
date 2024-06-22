# Imports
import os
import csv
import math
from turtle import speed
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import tifffile
import json
import re
from tqdm.auto import tqdm
from collections import defaultdict
from multiprocessing import Process, cpu_count
import concurrent.futures
from itertools import islice
from fractions import Fraction
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from scipy.stats import chisquare
from scipy.stats import entropy
from scipy.special import kl_div
from sklearn.cluster import KMeans
from scipy import stats
from skimage import io
import cv2

# cd C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Experiments
# python3 .\preprocess_existing.py

def fill_pixel(tol, pixel_x, pixel_z, intensity, image, resolution):
    left = int(max(pixel_x-tol,0))
    right = int(min(pixel_x+tol,resolution))
    top = int(min(pixel_z+tol,resolution))
    bottom = int(max(pixel_z-tol,0))
    image[left:right,bottom:top] = intensity
    return image

def skiprows(index):
    if index < 0:
        return True
    else:
        return (index % 10 != 0)
    
def read_csv_files(csv_directory):
    csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]

    data_dict = {}
    all_dfs = []

    column_names = ['timestep','pos_x', 'pos_z']

    row_threshold = 3
    max_x = []
    min_x = []
    min_z = []
    max_z = []
    for filename in tqdm(csv_files): #TODO
        # Read the CSV file into a pandas DataFrame and assign column names
        # df = pd.read_csv(os.path.join(csv_directory, filename), 
        #     header=None, names=column_names, 
        #     skiprows=None,
        #     #skiprows=lambda index: index == 0 or skip_rows(index, row_step),
        #     usecols=[0, 1, 2])
        df = pd.read_csv(os.path.join(csv_directory, filename), header=None, skiprows= skiprows)
        df[column_names] = df[0].str.split(';', expand=True)
        df[column_names[0]] = df[column_names[0]].astype(float) 
        df[column_names[1]] = df[column_names[1]].astype(float)
        df[column_names[2]] = df[column_names[2]].astype(float)
        df.drop(0, axis=1, inplace=True)
        if df.shape[0] < row_threshold:
            continue
        df["speed"] = 0
        for i in range(1, len(df)):
            df.loc[i, "speed"] = math.sqrt((df.loc[i, 'pos_x'] - df.loc[i - 1, 'pos_x']) ** 2 + (df.loc[i, 'pos_z'] - df.loc[i - 1, 'pos_z']) ** 2)

        min_x.append(min(df["pos_x"]))
        max_x.append(max(df["pos_x"]))
        min_z.append(min(df["pos_z"]))
        max_z.append(max(df["pos_z"]))


        # Maximum supported duration 15secs and minimum 6secs.
        start_time = df["timestep"].iloc[0]
        end_time =  df["timestep"].iloc[-1] - start_time
        if end_time < 6:
            continue
        elif end_time > 15:
            thersh = start_time + 15
            df1 = df[df['timestep'] <= thersh].copy()
            df1.drop("timestep", axis=1, inplace=True)
            filename1 = filename.split('.csv')[0] + "_1" + ".csv"
            data_dict[filename1] = df1
            all_dfs.append(df1)

            if (end_time - 15) >= 6:
                df2 = df[df['timestep'] > thersh].copy()
                df2.drop("timestep", axis=1, inplace=True)
                filename2 = filename.split('.csv')[0] + "_2" + ".csv"
                data_dict[filename2] = df2
                all_dfs.append(df2)
        else:
            df.drop("timestep", axis=1, inplace=True)
            data_dict[filename] = df
            all_dfs.append(df)
    
    # Specific source assumption for Arxiepiskopi:
    maxX = np.max(max_x)
    minX = np.min(min_x)
    maxZ = np.max(max_z)
    minZ = np.min(min_z)
    source_x = (minX + maxX)/2
    source_z = minZ - 1
    # Specific source assumption for Zara03:
    # source_x = 0.6
    # source_z = 0.15

    for filename, df in data_dict.items():
        # Normalize to [0, 1]
        bound_min = min(np.min(df["pos_x"]), np.min(df["pos_z"]), source_x, source_z) # Source is no longer at (0,0)
        bound_max = max(np.max(df["pos_x"]), np.max(df["pos_z"]), source_x, source_z)

        bound_max += 1.5
        bound_min -= 1.5

        df["pos_x"] = (df['pos_x'] - bound_min) / (bound_max - bound_min) * (1 - 0) 
        df["pos_z"] = (df['pos_z'] - bound_min) / (bound_max - bound_min) * (1 - 0) 


        s = len(df["pos_x"])
        # print(filename, s)
        source_norm = np.zeros((s)) #TODO not necessarily at 0
        source_norm[0] = (source_x - bound_min) / (bound_max - bound_min) * (1 - 0)
        source_norm[1] = (source_z - bound_min) / (bound_max - bound_min) * (1 - 0)
        df["norm_source"] = list(source_norm)

        data_dict[filename] = df

    return data_dict

def read_csv_files_other(csv_directory):

    # 2.5fps => min 6s = 15frames & max 15sec = 37.5

    traj_array = np.loadtxt(csv_directory)
    column_names = ["Frame_No", "Pedestrian_ID", "pos_x", "pos_z", "pos_y", "v_x", "v_z", "v_y" ]
    df = pd.DataFrame(traj_array, columns = column_names)
    df = df.drop("pos_z", axis = 1)
    df = df.drop("v_z", axis = 1)
    df = df.drop("v_x", axis = 1)
    df = df.drop("v_y", axis = 1)
    num_agents = int(df["Pedestrian_ID"].max())

    data_dict = {}
    all_dfs = []

    for agent_id in range(1,num_agents+1):
        key = "agent_" + str(agent_id)

        agent_traj = df[df["Pedestrian_ID"] == agent_id]
        agent_traj = agent_traj.drop("Pedestrian_ID", axis = 1)
        agents_traj_values = agent_traj.values
        # Remove non existent agent IDs
        if len(agents_traj_values) == 0:
          continue
        frame_cutoff = agents_traj_values[-1,0] 
        start_frame = agents_traj_values[0,0]
        if (frame_cutoff - start_frame) < 16:
            continue
        if (frame_cutoff - start_frame) >= 38:
            agents_traj_values = agents_traj_values[:4,:]
        cutoff = 0
        cutoff_list = []
        start_list = [0]
        for i in range(1,agents_traj_values.shape[0]):
          end_frame = agents_traj_values[i,0]
          if end_frame-start_frame > frame_cutoff:
            cutoff += 1
            cutoff_list.append(i)
            start_list.append(i)
            start_frame = end_frame
        cutoff_list.append(agents_traj_values.shape[0]+1)

        for c in range(cutoff+1):
          all_dfs.append(agents_traj_values[start_list[c]:cutoff_list[c],:])
          key_up = key + "_" +str(c)
          data_dict[key_up] = agents_traj_values[start_list[c]:cutoff_list[c],:]

    source_x = 0
    source_z = 0

    for filename, arr in data_dict.items():
        # Normalize to [0, 1]
        bound_min = min(np.min(arr[:,1]), np.min(arr[:,2]), source_x, source_z)
        bound_max = max(np.max(arr[:,1]), np.max(arr[:,2]),  source_x, source_z)

        bound_max += 0.7
        bound_min -= 0.7 

        arr[:,1] = (arr[:,1] - bound_min) / (bound_max - bound_min) * (1 - 0) 
        arr[:,2] = (arr[:,2] - bound_min) / (bound_max - bound_min) * (1 - 0) 


        s = len(arr[:,1])
        source_norm = np.zeros((s)) 
        source_norm[0] = (source_x - bound_min) / (bound_max - bound_min) * (1 - 0)
        source_norm[1] = (source_z - bound_min) / (bound_max - bound_min) * (1 - 0)
        # df["norm_source"] = list(source_norm)

        data_dict[filename] = arr

    return data_dict

def create_images(key, value, dataset_name, resolution= 32):
    plt.plot(value["pos_x"].to_numpy(), value["pos_z"].to_numpy(), '.')
    pixel_pos_x = value["pos_x"].to_numpy() * (resolution - 1)
    pixel_pos_z = value["pos_z"].to_numpy() * (resolution - 1)
    image = np.zeros((resolution,resolution), np.float32)
    source_pos_x = value["norm_source"].iloc[0] * (resolution - 1)
    source_pos_z = value["norm_source"].iloc[1] * (resolution - 1)
    same_speed_count = 0
    for i in range(len(pixel_pos_x)):
        pixel_x = int(pixel_pos_x[i])
        pixel_z = int(pixel_pos_z[i])
        if i == 0:
            pixel_x_init = pixel_x
            pixel_z_init = pixel_z
            image[pixel_x,pixel_z] = 1
        elif (value["speed"].iloc[i] <= 0.001): 
            same_speed_count += 1

        cur_speed = (1- value["speed"].iloc[i])*0.6
        if same_speed_count >= 5:
            tol = 2
            left = int(max(pixel_x-tol,0))
            right = int(min(pixel_x+tol,resolution))
            top = int(min(pixel_z+tol,resolution))
            bottom = int(max(pixel_z-tol,0))
            image[left:right,bottom:top] = cur_speed
            same_speed_count = 0
        else:
            image[pixel_x,pixel_z] = cur_speed

    image[pixel_x_init,pixel_z_init] = 1
    image = fill_pixel(1, pixel_x_init, pixel_z_init, 1, image, resolution)
    # tifffile.imwrite(dataset_name + "\\" + key + '_s' + '.tif', image)

    # Place source 
    image[int(source_pos_x), int(source_pos_z)] = 1 # TODO decide whether to include source in image.
    image = fill_pixel(1, int(source_pos_x), int(source_pos_z), 1, image, resolution)
    tifffile.imwrite(dataset_name + "\\" + key + '.tif', image)

def generate_python_files(folder_path, name):

    all_files = os.listdir(folder_path)
    tif_files = [file for file in all_files if file.lower().endswith('.tif')]


    for tif_file in tqdm(tif_files):
        old_name = tif_file.split('.')[0]
        try:
            image_path = os.path.join(folder_path, tif_file)
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            # tifffile.imwrite(f'C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\PythonFiles\{name}\{old_name}.tif', image)
            np.savez(f'C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\PythonFiles\{name}\{old_name}.npz', image)
        except Exception as e:
            print(f"Error loading image '{tif_file}': {e}")

def existing_data_preprocessing(current_file_dir, name):
    csv_directory  = current_file_dir + name + "\\"
    # csv_directory  = current_file_dir + name

    csv_data = read_csv_files(csv_directory)
    n_csvs = len(csv_data)
    dict_list = list(csv_data.items())

    # key, value = dict_list[20]
    # print(key, value)
    # exit()

    for i in tqdm(range(n_csvs)):
        key, value = dict_list[i]
        prefix = key.split(".")[0]
        folder_path = "C:\\PROJECTS\\SocialLandmarks\\SocialLandmarks_Python\\Data\\Images" + name
        # folder_path = "C:\\PROJECTS\\SocialLandmarks\\SocialLandmarks_Python\\Data\\Images" + "\ETH"
        # dataset_name = name
        files = os.listdir(folder_path)
        file_exists = any(file.startswith(prefix) for file in files)
        if file_exists == False:
            empty_predictions = create_images(prefix, value, folder_path)
    
    plt.show()

    generate_python_files(folder_path, name)
    print("DONE! Preprocessing Successful.")

# Execute
if __name__ ==  '__main__':
    current_file_dir = "C:\PROJECTS\SocialLandmarks\Data\Trajectories"
    # name = "\Zara\Zara03"
    name = "\Flock"
    # name = "\Students\Students01"
    # name = "\eth_hotel.txt" # TODO
    
    existing_data_preprocessing(current_file_dir, name)
