# IMPORTS
import numpy as np
from PIL import Image
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from tqdm import tqdm
import pickle
import torch.utils.data as data
import tifffile

# python3 C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\SingleSwitch\preprocess_model_data.py

# HELPER FUNCTIONS
def zoom_image(image):
    min_zoom = 1.1
    max_zoom = 2
    random_zoom = np.random.uniform(min_zoom, max_zoom)
    center_x, center_y = width // 2, height // 2
    scaled_width, scaled_height = int(width * random_zoom), int(height * random_zoom)
    zoom_matrix = np.array([[random_zoom, 0, (1 - random_zoom) * center_x],
                            [0, random_zoom, (1 - random_zoom) * center_y]], dtype=np.float32)
    image_zoomed = cv2.warpAffine(image, zoom_matrix, (width, height), flags=cv2.INTER_LINEAR)
    return image_zoomed
def normalize(image):
    mean = np.mean(image)
    std = np.std(image)
    scaled_image = (image - mean) / std
    return scaled_image
def add_gaussian_noise(image, mean=0, std=0.06):
    row, col = image.shape
    gauss = np.random.normal(mean, std, (row, col))
    noisy = np.clip(image + gauss, 0, 255)
    return noisy.astype(np.float32)

substring = "_a1"
counter = 0

# # TEST DATA:
# folder_path_3 = 'C:/PROJECTS/SocialLandmarks/SocialLandmarks_Python/Data/Images/NoLandmarks/TestData' 
# all_files = os.listdir(folder_path_3)
# tif_files = [file for file in all_files if file.lower().endswith('.tif')]
# for tif_file in tqdm(tif_files):
#     old_name = tif_file.split(substring, 1)[0]
#     try:
#         counter += 1
#         image_path = os.path.join(folder_path_3, tif_file)
#         image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
#         #landmark_type.append(2) #2: No Landmark
#         np.savez(f'C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\PythonFiles\SingleSwitch\TestData\{old_name}_{counter}_type_2_rot_0.npz', image)
#     except Exception as e:
#         print(f"Error loading image '{tif_file}': {e}")
# print("SIMPLE TEST IMAGES LOADED!")

# exit()

# DATA PRE-PROCESSING:
folder_path = 'C:/PROJECTS/SocialLandmarks/SocialLandmarks_Python/Data/Images/SingleSwitch_Trial2/TestData' 
all_files = os.listdir(folder_path)

tif_files = [file for file in all_files if file.lower().endswith('.tif')]
for tif_file in tqdm(tif_files):
    old_name = tif_file.split(substring)[0] + tif_file.split(substring)[1].split(".tif")[0]
    # field_1 = tif_file.split("IF_")[1].split("_")[0]
    # field_2 = tif_file.split("IF_")[1].split("_")[1]
    try:
        counter += 1
        image_path = os.path.join(folder_path, tif_file)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        np.savez(f'C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\PythonFiles\SingleSwitch\TestData\{old_name}_{counter}_rot_0.npz', image)
        # tifffile.imwrite("C:\\PROJECTS\\SocialLandmarks\\SocialLandmarks_Python\\Data\\PythonFiles\SingleSwitch\\unnormalized.tif", image)
        # np.savez(f'C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\PythonFiles\SingleSwitch\{old_name}_{counter}_norm_type_{class_id}_rot_0.npz', normalize(image))
        # np.savez(f'C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\PythonFiles\SingleSwitch\{old_name}_{counter}_noisy_rot_0.npz', add_gaussian_noise(normalize(image)))
        # tifffile.imwrite("C:\\PROJECTS\\SocialLandmarks\\SocialLandmarks_Python\\Data\\PythonFiles\SingleSwitch\\normal.tif",  normalize(image))
        # tifffile.imwrite("C:\\PROJECTS\\SocialLandmarks\\SocialLandmarks_Python\\Data\\PythonFiles\SingleSwitch\\noisy.tif",  add_gaussian_noise(normalize(image)))
        height, width = image.shape[:2]
        min_angle = -179
        max_angle = 179
        # random_angles = np.random.uniform(min_angle, max_angle, size=5) #size was 10
        random_angles = [90,180,270]
        for i, angle in enumerate(random_angles):
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image_rotated = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)
            # TODO np.savez(f'C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\PythonFiles\SingleSwitch\{old_name}_{counter}_rot_{int(angle)}.npz', image_rotated)
            # tifffile.imwrite(f"C:\\PROJECTS\\SocialLandmarks\\SocialLandmarks_Python\\Data\\PythonFiles\SingleSwitch\\imgrot_{angle}.tif", image_rotated)
            # np.savez(f'C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\PythonFiles\SingleSwitch\{old_name}_{counter}_norm_type_2_rot_{int(angle)}.npz', normalize(image_rotated))
            # np.savez(f'C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\PythonFiles\SingleSwitch\{old_name}_{counter}_rot_{int(angle)}.npz', add_gaussian_noise(normalize(image_rotated)))
    except Exception as e:
        print(f"Error loading image '{tif_file}': {e}")
print("SingleSwitch BEHAVIOR IMAGES LOADED!")

exit()