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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

os.environ['WANDB_API_KEY']="29162836c3095b286c169bf889de71652ed3201b"

#TODO: go to cd C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data
# Execute python3 C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\SingleSwitch\train_model2.py

# HELPER FUNCTIONS
def visualize_image(array):
    image_rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    # plt.imshow(array, cmap='gray')
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

# Main functions:
def load_data(check = False):
    folder_path = 'PythonFiles\\SingleSwitch\\'  
    file_list = os.listdir(folder_path)
    npz_files = [file for file in file_list if file.endswith('.npz')]
    loaded_images = []
    f_field_IDs = []
    s_field_IDs = []
    raw_gt = []
    gt = []
    # Initialize first InF classes:
    f_class1 = 0
    f_class2 = 0
    f_class3 = 0
    f_class4 = 0
    f_class5 = 0
    f_class6 = 0
    # Initialize second InF classes:
    s_class1 = 0
    s_class2 = 0
    s_class3 = 0
    s_class4 = 0
    s_class5 = 0
    s_class6 = 0
    for npz_file in tqdm(npz_files):
        # Read gt fields:
        class_index = npz_file.split("IF_")[1].split("_T")[0]
        field_1 = int(npz_file.split("IF_")[1].split("_")[0])
        field_2 = int(npz_file.split("IF_")[1].split("_")[1])
        raw_gt.append(class_index)

        # Read image:
        file_path = os.path.join(folder_path, npz_file)
        loaded_data = np.load(file_path)
        array_keys = loaded_data.files
        array_key = array_keys[0]
        array = loaded_data[array_key]
        if (array.dtype != 'float32'):
            print("ERROR! Check file path: ",file_path)
            exit()
        loaded_images.append(array)        

        # Assign examples to classes:
        if field_1 == 0:
            field1_int = 3
            f_class3 += 1
        elif field_1 == 1:
            field1_int = 1
            f_class1 += 1
        elif field_1 == 2:
            field1_int = 2
            f_class2 += 1
        elif field_1 == 3:
            field1_int = 3
            f_class3 += 1
        elif field_1 == 4:
            field1_int = 4
            f_class4 += 1
        elif field_1 == 5:
            field1_int = 5
            f_class5 += 1
        elif field_1 == 6:
            field1_int = 6
            f_class6 += 1
        else:
            print("ERROR: Unrecognised 1st Interaction Field: ", field_1)
            exit()
        f_field_IDs.append(field1_int)
        if field_2 == 0:
            sfield2_int = 3
            s_class3 += 1
        elif field_2 == 1:
            field2_int = 1
            s_class1 += 1
        elif field_2 == 2:
            field2_int = 2
            s_class2 += 1
        elif field_2 == 3:
            field2_int = 3
            s_class3 += 1
        elif field_2 == 4:
            field2_int = 4
            s_class4 += 1
        elif field_2 == 5:
            field2_int = 5
            s_class5 += 1
        elif field_2 == 6:
            field2_int = 6
            s_class6 += 1
        else:
            print("ERROR: Unrecognised 2nd Interaction Field: ", field_2)
            exit()
        s_field_IDs.append(field2_int)
        gt.append(f'{field1_int}_{field2_int}')
        
    if check == True:
        print("1st Interaction Field:")
        print(f_class1, f_class2, f_class3, f_class4, f_class5, f_class6)
        # 12288 11717 11493 11398 12464 11974
        print("2nd Interaction Field:")
        print(s_class1, s_class2, s_class3, s_class4, s_class5, s_class6)
        # 11728 12087 12094 11864 11664 11897

    return loaded_images, f_field_IDs, s_field_IDs, gt

# Classes:
class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        # first_field = torch.Tensor(int(label.split("_")[0]))
        # second_field = torch.Tensor(int(label.split("_")[1]))

        image = torch.from_numpy(image)

        return image, label

if __name__ ==  '__main__':
    """
    Train Pytorch model to predict separately the 1st and 2nd InF.
    Preprocessing:
        Load the npy files and build gt (x,y) pairs.
    Inputs:
        Model inputs are (32,32,1) # TODO:check trajectory images.
    Outputs:
        Model outputs are predictions of 1st and 2nd InF.
        The code saves the trained model,
        and the confusion matrices. #TODO: verify
    """

    # Define global parameters:
    batch_size = 32

    # Preprocess data:
    loaded_images, f_InFs, s_InFs, gt = load_data()
    # i = 12000 print(gt[i: (i+10)], f_InFs[i: (i+10)], s_InFs[i: (i+10)])
    images = scale_to_standard_normal(loaded_images) # TODO:PC1
    print("Loaded Data: ", images.shape, len(gt))
    # index = 6 visualize_image(loaded_images[index]) visualize_image(images[index]) exit()
    x_train, x_val, y_train, y_val = train_test_split(images, gt, test_size=0.2, random_state=42)
    train_dataset = CustomDataset(x_train, y_train)
    val_dataset = CustomDataset(x_val, y_val)
    trainloader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valoader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    # for i, (inputs, labels) in enumerate(trainloader):
    #     print(inputs.shape)
    #     print(labels)
    #     exit()

exit()



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second Convolutional Layer: input size (16x16x16), output size (8x8x32)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Fully connected layers
        self.fc1 = nn.Linear(8*8*32, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.relu5 = nn.ReLU()
        self.fc4 = nn.Linear(64, 36)  # Output layer with 36 classes

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Flatten the output of the last convolutional layer
        x = self.flatten(x)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        x = self.relu5(x)
        x = self.fc4(x)
        
        return x
        
        return x

# Instantiate the model
model = CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# inputs= torch.randn(batch_size, 1, 32, 32, requires_grad=True)
# #inputs = np.random.random((batch_size, 32, 32, 1))
# print(inputs.shape)
# outputs =  model(inputs)
# print(outputs.shape)
# exit()

# HYPERPARAMETERS
wandb.init(project="SocialLandmarks")
config = wandb.config
config.epochs = 50
config.batch_size = batch_size


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, train_loader, criterion, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.unsqueeze(1).to(device), labels.long().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i + 1) % 1 == 0:
                #print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / (i + 1):.4f}')
                
                # Log metrics to wandb
                wandb.log({"epoch": epoch+1, "step": i+1, "loss": running_loss / (i + 1)})
    print('Finished Training')

# Train the model
wandb.watch(model, log_freq=1)
train(model, train_loader, criterion, optimizer, device, config.epochs)

# model.save("C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\SingleSwitch\exp_2.h5")
torch.save(model.state_dict(), "C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\SingleSwitch\exp_2.h5")
print("MODEL IS SAVED!!")
wandb.finish()

# # CONFUSION MATRICES:

# y_train_pred = model.predict(x_train)
# y_train_pred_classes = y_train_pred.argmax(axis=-1)
# confusion = confusion_matrix(y_train, y_train_pred_classes)
# print("Confusion Matrix for Training Data:")
# print(confusion)
# print(np.max(confusion), np.argmax(confusion))

# y_val_pred = model.predict(x_val)
# y_val_pred_classes = y_val_pred.argmax(axis=-1)
# confusion = confusion_matrix(y_val, y_val_pred_classes)
# print("Confusion Matrix for Validation Data:")
# print(confusion)
# print(np.max(confusion), np.argmax(confusion))

