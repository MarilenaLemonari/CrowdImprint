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
import random

#TODO: remove source csv.
# cd C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data
# python3 .\generate_trajectory_images.py

def fill_pixel(tol, pixel_x, pixel_z, intensity, image, resolution):
    left = int(max(pixel_x-tol,0))
    right = int(min(pixel_x+tol,resolution))
    top = int(min(pixel_z+tol,resolution))
    bottom = int(max(pixel_z-tol,0))
    image[left:right,bottom:top] = intensity
    return image

def read_csv_files(csv_directory):
    csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]

    data_dict = {}
    all_dfs = []
    column_names = ['timestep','pos_x', 'pos_z']

    row_threshold = 3
    for filename in tqdm(csv_files):
        # Read the CSV file into a pandas DataFrame and assign column names
        df = pd.read_csv(os.path.join(csv_directory, filename), 
            header=None, names=column_names, 
            skiprows=None,
            #skiprows=lambda index: index == 0 or skip_rows(index, row_step),
            usecols=[0, 1, 2])
        df[column_names[0]] = df[column_names[0]].astype(float) 
        df[column_names[1]] = df[column_names[1]].astype(float)
        df[column_names[2]] = df[column_names[2]].astype(float)
        # df.drop(0, axis=1, inplace=True)
        if df.shape[0] < row_threshold:
            continue

        df["speed"] = 0
        for i in range(1, len(df)):
            df.loc[i, "speed"] = math.sqrt((df.loc[i, 'pos_x'] - df.loc[i - 1, 'pos_x']) ** 2 + (df.loc[i, 'pos_z'] - df.loc[i - 1, 'pos_z']) ** 2)

        df.drop("timestep", axis=1, inplace=True)
        data_dict[filename] = df
        all_dfs.append(df)
    
    for filename, df in data_dict.items():
        # Normalize to [0, 1]
        bound_min = min(np.min(df["pos_x"]), np.min(df["pos_z"]), 0)
        bound_max = max(np.max(df["pos_x"]), np.max(df["pos_z"]), 0)

        bound_max += 0.5
        bound_min -= 0.5 

        df["pos_x"] = (df['pos_x'] - bound_min) / (bound_max - bound_min) * (1 - 0) 
        df["pos_z"] = (df['pos_z'] - bound_min) / (bound_max - bound_min) * (1 - 0) 


        s = len(df["pos_x"])
        source_norm = np.zeros((s))
        source_norm[0] = (0 - bound_min) / (bound_max - bound_min) * (1 - 0)
        df["norm_source"] = list(source_norm)

        data_dict[filename] = df

    return data_dict

def create_images(key, value, dataset_name, resolution= 32):
    # default_int = 0.5
    # plt.clf()
    # plt.plot(value["pos_x"], value["pos_z"], c = 'slategrey')
    # plt.scatter(value["pos_x"][0], value["pos_z"][0], c = 'slategrey')
    # plt.scatter(value["norm_source"][0], value["norm_source"][0], c = 'firebrick', marker = '*', s = 200)
    # plt.legend(['Path', 'Spawn', 'Source'])
    # plt.xlabel("Position X")
    # plt.ylabel("Position Z")
    # plt.title("Normalized Path Image")
    # plt.savefig(dataset_name + "\\" + key)
    pixel_pos_x = value["pos_x"] * (resolution - 1)
    pixel_pos_z = value["pos_z"] * (resolution - 1)
    image = np.zeros((resolution,resolution), np.float32)
    source_pos = value["norm_source"][0] * (resolution - 1)
    same_speed_count = 0
    for i in range(len(pixel_pos_x)):
        pixel_x = int(pixel_pos_x[i])
        pixel_z = int(pixel_pos_z[i])
        if i == 0:
            pixel_x_init = pixel_x
            pixel_z_init = pixel_z
            image[pixel_x,pixel_z] = 1
        elif (value["speed"][i] <= 0.001): # ML1: value["speed"][i-1]
            same_speed_count += 1

        cur_speed = (1- value["speed"][i])*0.6
        if same_speed_count >= 5:
            tol = 1
            left = int(max(pixel_x-tol,0))
            right = int(min(pixel_x+tol,resolution))
            top = int(min(pixel_z+tol,resolution))
            bottom = int(max(pixel_z-tol,0))
            image[left:right,bottom:top] = cur_speed
        else:
            image[pixel_x,pixel_z] = cur_speed


    image[pixel_x_init,pixel_z_init] = 1

    random_value = random.choice([0,1])
    if random_value == 0:
        tifffile.imwrite(dataset_name + "\\" + key + '_s' + '.tif', image)
    elif random_value == 1:
    # Place source 
        image[int(source_pos), int(source_pos)] = 1
        tifffile.imwrite(dataset_name + "\\" + key + '.tif', image)
    else:
        print("ERROR! wWrong random value.")
        exit()

def create_centrered_images(key, value, dataset_name, resolution= 32):
    # default_int = 0.5
    # plt.clf()
    # plt.plot(value["pos_x"], value["pos_z"], c = 'slategrey')
    # plt.scatter(value["pos_x"][0], value["pos_z"][0], c = 'slategrey')
    # plt.scatter(value["norm_source"][0], value["norm_source"][0], c = 'firebrick', marker = '*', s = 200)
    # plt.legend(['Path', 'Spawn', 'Source'])
    # plt.xlabel("Position X")
    # plt.ylabel("Position Z")
    # plt.title("Normalized Path Image")
    # plt.savefig(dataset_name + "\\" + key)
    max_diff = max(value["norm_source"][0], 1 - value["norm_source"][0])
    # transform from [0,1] to range where source pos is 0:
    source_pos = value["norm_source"][0] - value["norm_source"][0]
    pixel_pos_x = value["pos_x"] - value["norm_source"][0]
    pixel_pos_z = value["pos_z"] - value["norm_source"][0]
    # Zoom
    source_pos *= ((resolution - 1) / ( 2* max_diff))
    pixel_pos_x *= ((resolution - 1) / (2 * max_diff))
    pixel_pos_z *= ((resolution - 1) / (2 * max_diff))
    source_pos += ((resolution - 1)/2)
    pixel_pos_x += ((resolution - 1)/2)
    pixel_pos_z += ((resolution - 1)/2)
    image = np.zeros((resolution,resolution), np.float32)
    same_speed_count = 0
    for i in range(len(pixel_pos_x)):
        pixel_x = int(pixel_pos_x[i])
        pixel_z = int(pixel_pos_z[i])
        if i == 0:
            pixel_x_init = pixel_x
            pixel_z_init = pixel_z
            image[pixel_x,pixel_z] = 1
        elif (value["speed"][i] == value["speed"][i-1]):
            same_speed_count += 1

        cur_speed = (1- value["speed"][i])*0.6
        if same_speed_count >= 5:
            # tol = 1
            # left = int(max(pixel_x-tol,0))
            # right = int(min(pixel_x+tol,resolution))
            # top = int(min(pixel_z+tol,resolution))
            # bottom = int(max(pixel_z-tol,0))
            # image[left:right,bottom:top] = cur_speed
            image = fill_pixel(1, pixel_x, pixel_z, cur_speed, image, resolution)
        else:
            image[pixel_x,pixel_z] = cur_speed


    image[pixel_x_init,pixel_z_init] = 1
    image = fill_pixel(1, pixel_x_init, pixel_z_init, 1, image, resolution)
    # tifffile.imwrite(dataset_name + "\\" + key + '_s' + '.tif', image)

    # Place source 
    image[int(source_pos), int(source_pos)] = 1
    image = fill_pixel(1, int(source_pos), int(source_pos), 1, image, resolution)
    tifffile.imwrite(dataset_name + "\\" + key + '.tif', image)

# Execute
if __name__ ==  '__main__':
    current_file_dir = "C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\Trajectories"
    name = "\SingleSwitch\TestData" #TODO
    # name = "\\NoSwitch"
    
    csv_directory  = current_file_dir + name + "\\"

    csv_data = read_csv_files(csv_directory)
    n_csvs = len(csv_data)
    dict_list = list(csv_data.items())

    # key, value = dict_list[10]
    # print(key, value)
    # exit()

    for i in tqdm(range(n_csvs)):
        key, value = dict_list[i]
        prefix = key.split(".")[0]
        folder_path = "C:\\PROJECTS\\SocialLandmarks\\SocialLandmarks_Python\\Data\\Images" + name

        # plt.plot(value["pos_x"], value["pos_z"], 'slategrey')
        # plt.scatter(value["pos_x"][0], value["pos_z"][0], c = 'slategrey')
        # plt.scatter(value["norm_source"][0], value["norm_source"][0], c='firebrick', marker='*', s = 200)
        # plt.legend(['Path', 'Spawn', 'Source'])
        # plt.xlabel("Position X")
        # plt.ylabel("Position Z")
        # plt.title("Raw Path Image")
        # plt.savefig(folder_path + "\\" + prefix)
        # plt.clf()

        # dataset_name = name
        files = os.listdir(folder_path)
        file_exists = any(file.startswith(prefix) for file in files)
        if file_exists == False:
            empty_predictions = create_centrered_images(prefix, value, folder_path)

    print("DONE! Preprocessing Successful.")
