# IMPORTS
from importlib.metadata import requires
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import torch.utils.data as data
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import matplotlib.pyplot as plt
import wandb
from wandb.keras import WandbCallback
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from keras import backend as K
from tensorflow.keras import regularizers

os.environ['WANDB_API_KEY']="29162836c3095b286c169bf889de71652ed3201b"

#TODO: go to cd C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data
# Execute python3 C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\SingleSwitch\train_model1.py

# HELPER FUNCTIONS
def visualize_image(array):
    image_rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title('Image')
    plt.axis('off')
    plt.show()
def scale_to_standard_normal(images):
    mean = np.mean(images)
    std = np.std(images)
    scaled_images = (images - mean) / std
    return scaled_images
def accuracy_first(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true[:, 0]), K.round(y_pred[:, 0])), axis=-1)
def accuracy_second(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true[:, 1]), K.round(y_pred[:, 1])), axis=-1)

# LOAD DATA
folder_path = 'PythonFiles\\SingleSwitch\\'  
file_list = os.listdir(folder_path)
npz_files = [file for file in file_list if file.endswith('.npz')]
loaded_images = []
field_IDs = []
gt = []
gt_dict = {"0_0": 0, "0_1": 1, "0_2": 2, "0_3": 3, "0_4": 4, "0_5": 5,
           "1_0": 6, "1_1": 7, "1_2": 8, "1_3": 9, "1_4": 10, "1_5": 11, 
           "2_0": 12, "2_1": 13, "2_2": 14, "2_3": 15, "2_4": 16, "2_5": 17,
            "3_0": 18, "3_1": 19, "3_2": 20,"3_3": 21, "3_4": 22, "3_5": 23,
            "4_0": 24, "4_1": 25, "4_2": 26, "4_3": 27, "4_4": 28, "4_5": 29,
            "5_0": 30, "5_1": 31, "5_2": 32, "5_3": 33, "5_4": 34, "5_5": 35}
class_0 = 0
class_1 = 0
class_2 = 0
class_3 = 0
class_4 = 0
class_5 = 0
class_6 = 0
class_7 = 0
class_8 = 0
class_9 = 0
class_10 = 0
class_11 = 0
class_12 = 0
class_13 = 0
class_14 = 0
class_15 = 0
class_16 = 0
class_17 = 0
class_18 = 0
class_19 = 0
class_20 = 0
class_21 = 0
class_22 = 0
class_23 = 0
class_24 = 0
class_25 = 0
class_26 = 0
class_27 = 0
class_28 = 0
class_29 = 0
class_30 = 0
class_31 = 0
class_32 = 0
class_33 = 0
class_34 = 0
class_35 = 0
for npz_file in tqdm(npz_files):
    # name_parts = npz_file.split('_')
    # type_index = name_parts.index("type")
    class_index = npz_file.split("IF_")[1].split("_T")[0]
    field_1 = npz_file.split("IF_")[1].split("_")[0]
    field_2 = npz_file.split("IF_")[1].split("_")[1]
    fields = np.array([field_1, field_2],dtype = np.float32)
    # value_after_type = int(name_parts[type_index + 1])

    # Read image:
    file_path = os.path.join(folder_path, npz_file)
    loaded_data = np.load(file_path)
    array_keys = loaded_data.files
    array_key = array_keys[0]
    array = loaded_data[array_key]
    if (array.dtype != 'float32'):
        print(file_path)
        exit()
    loaded_images.append(array)
    field_IDs.append(fields)
    class_type = gt_dict[class_index]
    gt.append(class_type)

    if class_type == 0:
        class_0 += 1
    elif class_type == 1:
        class_1 += 1
    elif class_type == 2:
        class_2 += 1
    elif class_type == 3:
        class_3 += 1
    elif class_type == 4:
        class_4 += 1
    elif class_type == 5:
        class_5 += 1
    elif class_type == 6:
        class_6 += 1
    elif class_type == 7:
        class_7 += 1
    elif class_type == 8:
        class_8 += 1
    elif class_type == 9:
        class_9 += 1
    elif class_type == 10:
        class_10 += 1
    elif class_type == 11:
        class_11 += 1
    elif class_type == 12:
        class_12 += 1
    elif class_type == 13:
        class_13 += 1
    elif class_type == 14:
        class_14 += 1
    elif class_type == 15:
        class_15 += 1
    elif class_type == 16:
        class_16 += 1
    elif class_type == 17:
        class_17 += 1
    elif class_type == 18:
        class_18 += 1
    elif class_type == 19:
        class_19 += 1
    elif class_type == 20:
        class_20 += 1
    elif class_type == 21:
        class_21 += 1
    elif class_type == 22:
        class_22 += 1
    elif class_type == 23:
        class_23 += 1
    elif class_type == 24:
        class_24 += 1
    elif class_type == 25:
        class_25 += 1
    elif class_type == 26:
        class_26 += 1
    elif class_type == 27:
        class_27 += 1
    elif class_type == 28:
        class_28 += 1
    elif class_type == 29:
        class_29 += 1
    elif class_type == 30:
        class_30 += 1
    elif class_type == 31:
        class_31 += 1
    elif class_type == 32:
        class_32 += 1
    elif class_type == 33:
        class_33 += 1
    elif class_type == 34:
        class_34 += 1
    elif class_type == 35:
        class_35 += 1
    else:
        print("Wrong Type: ",class_type)
        exit()


# print(class_0,class_1,class_2,class_3,class_4,class_5,class_6,class_7,class_8,class_9,class_10,class_11,class_12,class_13,class_14,class_15,
#       class_16, class_17, class_18, class_19, class_20, class_21, class_22, class_23, class_24, class_25, class_26, class_27, class_28, class_29, class_30,
#       class_31, class_32, class_33, class_34, class_35)
# # 326 332 312 318 364 308 332 314 350 330 336 316 348 338 340 316 338 360 328 340 358 354 330 348 320 348 342 354 308 338 294 340 328 354 312 326
# exit()

# NORMALIZE DATA
gt = np.array(gt)
#loaded_images = np.array(loaded_images)
x = scale_to_standard_normal(loaded_images)
# print(gt.shape, x.shape)
# exit()

# index =513
# print(loaded_images[index].shape)
# # visualize_image(loaded_images[index])
# print(landmark_type[index])
# # visualize_image(x[index])
# # exit()

# DATA LOADER
class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image = torch.from_numpy(image)

        return image, label

dataset = CustomDataset(x,gt)
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

x_train, x_val, y_train, y_val = train_test_split(x, gt, test_size=0.2, random_state=42)

model = Sequential()
# Convolutional layers
model.add(Conv2D(16, kernel_size=3, strides=1, padding='same', input_shape=(32, 32, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.1))
model.add(Conv2D(32, kernel_size=3, strides=1, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.1))
# Flatten the 3D feature maps to 1D
model.add(Flatten())

# Fully connected layers
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(36))
# model.add(Activation('sigmoid'))
model.add(Activation('softmax')) # TODO

# Creating an instance of the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

################################################################################################ REGULARIZATION MODEL
# model = Sequential()
# model.add(Conv2D(16, kernel_size=3, strides=1, padding='same', input_shape=(32, 32, 1))) 
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.1))
# model.add(Conv2D(32, kernel_size=3, strides=1, padding='same')) 
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.1))
# model.add(Flatten())
# model.add(Dense(1024, kernel_regularizer=regularizers.l2(0.01)))  
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128, kernel_regularizer=regularizers.l2(0.01)))  
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Dense(16))
# model.add(Activation('softmax'))
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
################################################################################################

# #inputs= torch.randn(batch_size, 32, 32,requires_grad=True)
# inputs = np.random.random((batch_size, 32, 32, 1))
# outputs =  model(inputs)
# print(outputs.shape)
# exit()

# HYPERPARAMETERS
wandb.init(project="SocialLandmarks")
config = wandb.config
config.epochs = 50
config.batch_size = batch_size

model.fit(x_train, y_train, epochs=config.epochs, batch_size=config.batch_size,
          validation_data=(x_val, y_val),
          callbacks=[WandbCallback()])
model.save("C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\SingleSwitch\exp_2.h5")
print("MODEL IS SAVED!!")
wandb.finish()

# CONFUSION MATRICES:

y_train_pred = model.predict(x_train)
y_train_pred_classes = y_train_pred.argmax(axis=-1)
confusion = confusion_matrix(y_train, y_train_pred_classes)
print("Confusion Matrix for Training Data:")
print(confusion)
print(np.max(confusion), np.argmax(confusion))

y_val_pred = model.predict(x_val)
y_val_pred_classes = y_val_pred.argmax(axis=-1)
confusion = confusion_matrix(y_val, y_val_pred_classes)
print("Confusion Matrix for Validation Data:")
print(confusion)
print(np.max(confusion), np.argmax(confusion))