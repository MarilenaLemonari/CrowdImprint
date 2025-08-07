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
import json

os.environ['WANDB_API_KEY']="29162836c3095b286c169bf889de71652ed3201b"

# TODO:
# cd C:\PROJECTS\SocialLandmarks
# Execute .\.venv\Scripts\activate 
# go to cd C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data
# Execute python3 C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\NoSwitch\train_noswitch_model.py

# HELPER FUNCTIONS
def visualize_image(array):
    image_rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    # plt.imshow(array, cmap='gray')
    plt.title('Image')
    plt.axis('off')
    plt.show()
def scale_to_standard_normal(images):
    mean = np.mean(images) # * 0
    std = np.std(images) # * 0 + 1
    scaled_images = (images - mean) / std
    return scaled_images
def accuracy_f(y_true, y_prob, correct, total):
    # Find y_pred:
    _, y_pred = torch.max(y_prob, 1)

    y_pred_r = y_pred + 1

    matches = (y_pred_r == y_true)
    correct += matches.sum().item()
    total += y_true.size(0)

    return correct, total, matches
def accuracy_overall(matches1, matches2):
    idx = [i for i in range(len(matches1)) if matches1[i] == matches2[i]]
    m = (matches1[idx] == True).sum().item()
    return m

# Main functions:
def load_data(check = False):
    folder_path = 'PythonFiles\\NoSwitch\\'  
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
    for npz_file in tqdm(npz_files): 
        # Read gt fields:
        class_index = npz_file.split("IF_")[1].split("_T")[0]
        field_1 = int(npz_file.split("IF_")[1].split("_")[0])
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
        gt.append(field1_int)
        
    if check == True:
        print("Interaction Field Class Balance:")
        print(f_class1, f_class2, f_class3, f_class4, f_class5, f_class6)
        # 11332 11810 12042 12163 12081 11855

    return loaded_images, f_field_IDs, gt
def validate(model, val_loader, criterion, device, CM = False):
    model.eval()
    val_loss = 0.0
    val_total = 0
    val_correct = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            fields = torch.Tensor(labels)
            if i == 0:
                y_val = labels
            else:
                y_val = torch.cat((y_val,labels), dim = 0)

            inputs = inputs.unsqueeze(1).to(device)
            fields = fields.long().to(device)

            field_prob = model(inputs)
            if i == 0:
                _, y_val_pred = torch.max(field_prob, 1) 
            else:
                _, y_pred = torch.max(field_prob, 1) 
                y_val_pred = torch.cat((y_val_pred, y_pred), dim = 0)

            loss = criterion(field_prob, fields - 1)
            val_loss += loss.item()

            correct1, total, matches1 = accuracy_f(fields, field_prob, 0, 0)
            val_correct += correct1
            val_total += total

    val_loss_avg = val_loss / len(val_loader)
    val_acc_overall_avg = val_correct / val_total

    if CM == True:
        confusion = confusion_matrix(y_val, y_val_pred + 1)
        # print(y_val)
        # print(y_val_pred)
        print("Confusion Matrix for Validation Data:")
        print(confusion)
        print(np.max(confusion), np.argmax(confusion))

    return val_loss_avg, val_acc_overall_avg
def train(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    model.train()
    epoch_losses = []
    acc_overall = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_total = 0
        correct = 0
        step_loss = []

        for i, (inputs, labels) in enumerate(train_loader):
            fields = torch.Tensor(labels)

            inputs = inputs.unsqueeze(1).to(device)
            fields = fields.long().to(device)

            optimizer.zero_grad()
            field_prob = model(inputs)
            loss= criterion(field_prob, fields - 1)
            loss.backward()
            optimizer.step()
            step_loss.append(loss.item())
            epoch_loss += loss.item()

            correct1, total, matches1 = accuracy_f(fields, field_prob, 0, 0)
            correct += correct1
            epoch_total += total

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}]:: Loss: [{epoch_loss / (i + 1):.4f}], Accuracy: {(correct * 100) /epoch_total:.2f}%')

                # Run validation
                val_loss, val_acc_overall = validate(model, val_loader, criterion, device)

                print(f'             Validation:: Loss: [{val_loss:.4f}],  Accuracy: {val_acc_overall*100:.2f}%')

                # # Log metrics to wandb
                # wandb.log({"epoch": epoch+1, "train_loss": epoch_loss / (i + 1), "tain_acc": correct /epoch_total,
                #             "val_loss": val_loss, "val_acc": val_acc_overall})

        epoch_losses.append(epoch_loss)
        acc_overall.append(correct / epoch_total)
        # Log metrics to wandb
        val_loss, val_acc_overall = validate(model, val_loader, criterion, device, CM = True)
        wandb.log({"epoch": epoch+1, "train_loss": epoch_loss/len(train_loader), "train_acc": correct /epoch_total,
                    "val_loss": val_loss, "val_acc": val_acc_overall})

        
    print('Finished Training')
    return epoch_losses, acc_overall

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
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers with Dropout
        self.fc1 = nn.Linear(8*8*32, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)  # Dropout layer with 50% dropout rate
        
        self.fc2 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.relu4 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)  # Dropout layer with 50% dropout rate

        self.fc3 = nn.Linear(128, 36)
        self.relu5 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.5)  # Dropout layer with 50% dropout rate
        
        # Output layers for two separate fields
        self.infd = nn.Linear(36, 6)

    def forward(self, x):
        # Convolutional layers with Batch Normalization
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Flatten the output of the last convolutional layer
        x = self.flatten(x)

        # Fully connected layers with Batch Normalization and Dropout
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        # x = self.relu5(x)
        # x = self.dropout3(x)
        # x = self.fc4(x)

        # Predictions
        field = self.infd(x)
        
        return field

if __name__ ==  '__main__':
    """
    Train Pytorch model to predict separately the 1st and 2nd InF.
    Preprocessing:
        Load the npy files and build gt (x,y) pairs.
    Inputs:
        Model inputs are (1,32,32) 
    Outputs:
        Model outputs are predictions of 1st and 2nd InF.
        The code saves the trained model,
        and the confusion matrices. #TODO: verify
    """

    # Define global parameters:
    batch_size = 32
    wandb.init(project="SocialLandmarks")
    config = wandb.config
    config.epochs = 20
    config.batch_size = batch_size

    # Preprocess data:
    loaded_images, f_InFs, gt = load_data()
    # i = 12000 print(gt[i: (i+10)], f_InFs[i: (i+10)], s_InFs[i: (i+10)])
    images = scale_to_standard_normal(loaded_images) # TODO:PC1
    print("Loaded Data: ", images.shape, len(gt))
    # index = 6 
    # visualize_image(loaded_images[index]) 
    # visualize_image(images[index]) 
    # exit()
    x_train, x_val, y_train, y_val = train_test_split(images, gt, test_size=0.2, random_state=42)
    train_dataset = CustomDataset(x_train, y_train)
    val_dataset = CustomDataset(x_val, y_val)
    trainloader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valoader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    # Instantiate the model
    model = CNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # inputs= torch.randn(batch_size, 1, 32, 32, requires_grad=True)
    # print(inputs.shape)
    # field =  model(inputs)
    # print(field.shape, field)
    # exit()

    # Model Parameters:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    wandb.watch(model, log_freq=1)
    epoch_losses, acc_overall = train(model, trainloader, valoader, criterion, optimizer, device, config.epochs)
    torch.save(model.state_dict(), "C:\\PROJECTS\\SocialLandmarks\\SocialLandmarks_Python\\Models\\NoSwitch\\test_noswitch_model.h5")
    print("MODEL IS SAVED!!")
    wandb.finish()

    training_metrics = {
    "epoch_losses": epoch_losses,
    "acc_overall": acc_overall
    }
    filename = 'training_metrics.json'
    with open(filename, 'w') as json_file:
        json.dump(training_metrics, json_file, indent=4)
    print(f"Data successfully written to {filename}")

    # # Confusion Matrices:
    # validate(model, trainloader, criterion, device, True) # Training
    # validate(model, valoader, criterion, device, True) # Validation