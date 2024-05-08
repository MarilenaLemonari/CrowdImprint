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
    
def read_csv_files(csv_directory, bound = 16):
    csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]
    # TODO: 10second trajectories.

    data_dict = {}
    all_dfs = []
    column_names = ['timestep','pos_x', 'pos_z']

    row_threshold = 3
    for filename in tqdm(csv_files):
        # Read the CSV file into a pandas DataFrame and assign column names
        # df = pd.read_csv(os.path.join(csv_directory, filename), 
        #     header=None, names=column_names, 
        #     skiprows=None,
        #     #skiprows=lambda index: index == 0 or skip_rows(index, row_step),
        #     usecols=[0, 1, 2])
        df = pd.read_csv(os.path.join(csv_directory, filename), header=None)
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


        data_dict[filename] = df
        all_dfs.append(df)
    
    # Make from [-12,12] to [0,1]
    for filename, df in data_dict.items():
        # Normalize to [0, 0.9]
        bound_min = min(np.min(df["pos_x"]), np.min(df["pos_z"]))
        bound_max = max(np.max(df["pos_x"]), np.max(df["pos_z"])) 

        df["pos_x"] = (df['pos_x'] - bound_min) / (bound_max - bound_min) * (0.9 - 0) 
        df["pos_z"] = (df['pos_z'] - bound_min) / (bound_max - bound_min) * (0.9 - 0) 


        data_dict[filename] = df
    return data_dict

def create_images(key, value, dataset_name, resolution= 32):
    # default_int = 0.5
    pixel_pos_x = value["pos_x"] * (resolution - 1)
    pixel_pos_z = value["pos_z"] * (resolution - 1)
    image = np.zeros((resolution,resolution), np.float32)
    # Place source 
    image[int(resolution/2), int(resolution/2)] = 1 #TODO
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
            tol = 1
            left = int(max(pixel_x-tol,0))
            right = int(min(pixel_x+tol,resolution))
            top = int(min(pixel_z+tol,resolution))
            bottom = int(max(pixel_z-tol,0))
            image[left:right,bottom:top] = cur_speed
        else:
            image[pixel_x,pixel_z] = cur_speed


    image[pixel_x_init,pixel_z_init] = 1

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
        # dataset_name = name
        files = os.listdir(folder_path)
        file_exists = any(file.startswith(prefix) for file in files)
        if file_exists == False:
            empty_predictions = create_images(prefix, value, folder_path)

    generate_python_files(folder_path, name)
    print("DONE! Preprocessing Successful.")

# Execute
if __name__ ==  '__main__':
    current_file_dir = "C:\PROJECTS\SocialLandmarks\Data\Trajectories"
    name = "\Flock"
    
    existing_data_preprocessing(current_file_dir, name)
