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
def validate(model, val_loader, criterion, device, CM = False):
    model.eval()
    val_loss = 0.0
    val_loss1 = 0.0
    val_loss2 = 0.0
    val_total = 0
    val_correct = 0
    val_acc_first = 0
    val_acc_sec = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            first_fields = []
            second_fields = []
            for label in labels:
                first_fields.append(int(label.split("_")[0]))
                second_fields.append(int(label.split("_")[1]))
            first_fields = torch.Tensor(first_fields)
            second_fields = torch.Tensor(second_fields)


            inputs = inputs.unsqueeze(1).to(device)
            first_fields = first_fields.long().to(device)
            second_fields = second_fields.long().to(device)

            field1_prob, field2_prob = model(inputs)

            if i == 0:
                y1_val = first_fields
                y2_val = second_fields
                _, y1_val_pred = torch.max(field1_prob, 1)
                _, y2_val_pred = torch.max(field2_prob, 1)
            else:
                y1_val = torch.cat((y1_val, first_fields), dim = 0)
                y2_val = torch.cat((y2_val, second_fields), dim = 0)
                _, y1_pred = torch.max(field1_prob, 1)
                _, y2_pred = torch.max(field2_prob, 1)
                y1_val_pred = torch.cat((y1_val_pred, y1_pred), dim = 0)
                y2_val_pred = torch.cat((y2_val_pred, y2_pred), dim = 0)

            loss_first = criterion(field1_prob, first_fields - 1)
            loss_second = criterion(field2_prob, second_fields - 1)
            loss = loss_first + loss_second
            val_loss += loss.item()
            val_loss1 += loss_first.item()
            val_loss2 += loss_second.item()

            correct1, total, matches1 = accuracy_f(first_fields, field1_prob, 0, 0)
            correct2, total, matches2 = accuracy_f(second_fields, field2_prob, 0, 0)
            val_correct += accuracy_overall(matches1, matches2)
            val_acc_first += correct1
            val_acc_sec += correct2
            val_total += total

    val_loss_avg = val_loss / len(val_loader)
    val_loss1_avg = val_loss1 / len(val_loader)
    val_loss2_avg = val_loss2 / len(val_loader)
    val_acc_first_avg = val_acc_first / val_total
    val_acc_sec_avg = val_acc_sec / val_total
    val_acc_overall_avg = val_correct / val_total

    if CM == True:
        # print(y1_val)
        # print(y1_val_pred)
        confusion1 = confusion_matrix(y1_val, y1_val_pred + 1)
        print("Validation Confusion Matrix for 1st:")
        print(confusion1)

        # print(y2_val)
        # print(y2_val_pred)
        confusion2 = confusion_matrix(y2_val, y2_val_pred + 1)
        print("Validation Confusion Matrix for 2nd:")
        print(confusion2)
        # print(np.max(confusion), np.argmax(confusion))

    return val_loss_avg, val_loss1_avg, val_loss2_avg, val_acc_first_avg, val_acc_sec_avg, val_acc_overall_avg
def train(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    model.train()
    epoch_losses = []
    epoch_accs_first = []
    epoch_accs_sec = []
    acc_overall = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_loss1 = 0.0
        epoch_loss2 = 0.0
        epoch_total = 0
        correct = 0
        epoch_acc_first = 0
        epoch_acc_sec = 0
        step_loss = []
        accuracy_first = []
        accuracy_second = []

        for i, (inputs, labels) in enumerate(train_loader):
            first_fields = []
            second_fields = []
            for label in labels:
                first_fields.append(int(label.split("_")[0]))
                second_fields.append(int(label.split("_")[1]))
            first_fields = torch.Tensor(first_fields)
            second_fields = torch.Tensor(second_fields)

            inputs = inputs.unsqueeze(1).to(device)
            first_fields = first_fields.long().to(device)
            second_fields = second_fields.long().to(device)

            optimizer.zero_grad()
            field1_prob, field2_prob = model(inputs)
            loss_first = criterion(field1_prob, first_fields - 1)
            loss_second = criterion(field2_prob, second_fields - 1)
            loss = loss_first + loss_second #TODO
            loss.backward()
            optimizer.step()
            step_loss.append(loss.item())
            epoch_loss += loss.item()
            epoch_loss1 += loss_first.item()
            epoch_loss2 += loss_second.item()

            correct1, total, matches1 = accuracy_f(first_fields, field1_prob, 0, 0)
            correct2, total, matches2 = accuracy_f(second_fields, field2_prob, 0, 0)
            correct += accuracy_overall(matches1, matches2)
            accuracy_first.append(correct1/total)
            accuracy_second.append(correct2/total)
            epoch_acc_first += correct1
            epoch_acc_sec += correct2
            epoch_total += total

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}]:: Loss 1: [{epoch_loss1 / (i + 1):.4f}], Loss 2: [{epoch_loss2 / (i + 1):.4f}], Loss: [{epoch_loss / (i + 1):.4f}], Accuracy 1st: {(epoch_acc_first * 100) /epoch_total:.2f}%, Accuracy 2nd: {(epoch_acc_sec * 100) /epoch_total:.2f}%, Accuracy: {(correct * 100) /epoch_total:.2f}%')

                # Run validation
                val_loss, val_loss1, val_loss2, val_acc_first, val_acc_sec, val_acc_overall = validate(model, val_loader, criterion, device)

                print(f'             Validation:: Loss 1: [{val_loss1:.4f}], Loss 2: [{val_loss2:.4f}], Loss: [{val_loss:.4f}], Accuracy 1st: {(val_acc_first*100):.2f}%, Accuracy 2nd: {val_acc_sec*100:.2f}%, Accuracy: {val_acc_overall*100:.2f}%')

                # # Log metrics to wandb
                # wandb.log({"epoch": epoch+1,"loss1": epoch_loss1 / (i + 1), 
                #            "loss2": epoch_loss2 / (i + 1), "train_loss": epoch_loss / (i + 1),
                #             "acc1": (epoch_acc_first) /epoch_total, "acc2": epoch_acc_sec /epoch_total, "train_acc": correct /epoch_total,
                #             "val_loss1": val_loss1, "val_loss2": val_loss2, "val_loss": val_loss, "val_acc1": val_acc_first, 
                #             "val_acc2": val_acc_sec, "val_acc": val_acc_overall})

        epoch_losses.append(epoch_loss)
        epoch_accs_first.append(epoch_acc_first / epoch_total)
        epoch_accs_sec.append(epoch_acc_sec / epoch_total)
        acc_overall.append(correct / epoch_total)

        val_loss, val_loss1, val_loss2, val_acc_first, val_acc_sec, val_acc_overall = validate(model, val_loader, criterion, device, CM = True)
        # Log metrics to wandb
        wandb.log({"epoch": epoch+1,"loss1": epoch_loss1 / len(train_loader), 
                    "loss2": epoch_loss2 / len(train_loader), "train_loss": epoch_loss / len(train_loader),
                    "acc1": (epoch_acc_first) /epoch_total, "acc2": epoch_acc_sec /epoch_total, "train_acc": correct /epoch_total,
                    "val_loss1": val_loss1, "val_loss2": val_loss2, "val_loss": val_loss, "val_acc1": val_acc_first, 
                    "val_acc2": val_acc_sec, "val_acc": val_acc_overall})

        
        
    print('Finished Training')
    return epoch_losses, epoch_accs_first, epoch_accs_sec, acc_overall

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

        # Common Convolutional Layer
        self.conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Layer for 1st
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # # Convolutional Layer for 2nd
        # self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.relu2 = nn.ReLU()
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers 1st
        self.fc1 = nn.Linear(8*8*32, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.8)
        self.fc2 = nn.Linear(256, 64)

        # # Fully connected layers 2nd
        # self.fc21 = nn.Linear(8*8*32, 256)
        # self.bn21 = nn.BatchNorm1d(256)
        # self.relu21 = nn.ReLU()
        # self.dropout21 = nn.Dropout(p=0.8)
        # self.fc22 = nn.Linear(256, 64)

        # Output layers for two separate fields
        # self.fist = nn.Linear(64, 6)
        # self.second = nn.Linear(64, 6)
        self.last = nn.Linear(64,12)

    def forward(self, x):
        # Convolutional layers with Batch Normalization
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Flatten the output of the last convolutional layer
        x = self.flatten(x)

        # Fully connected layers with Batch Normalization and Dropout
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)

        # Predictions
        # field1 = self.fist(x1)
        # field2 = self.second(x2)
        preds = self.last(x)
        field1 = preds[:,:6]
        field2 = preds[: , 6:]

        return field1, field2

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
    batch_size = 64
    wandb.init(project="SocialLandmarks")
    config = wandb.config
    config.epochs = 20
    config.batch_size = batch_size

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
    
    # Instantiate the model
    model = CNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # inputs= torch.randn(batch_size, 1, 32, 32, requires_grad=True)
    # print(inputs.shape)
    # field1, field2 =  model(inputs)
    # print(field1.shape, field2.shape)
    # exit()

    # Model Parameters:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) #, weight_decay=0.0001) # L2 regularization.

    # Train the model
    wandb.watch(model, log_freq=1)
    epoch_losses, epoch_accs_first, epoch_accs_sec, acc_overall = train(model, trainloader, valoader, criterion, optimizer, device, config.epochs)
    torch.save(model.state_dict(), "C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\SingleSwitch\\test_model.h5")
    print("MODEL IS SAVED!!")
    wandb.finish()

    training_metrics = {
    "epoch_losses": epoch_losses,
    "epoch_accs_first": epoch_accs_first,
    "epoch_accs_sec": epoch_accs_sec,
    "acc_overall": acc_overall
    }
    filename = 'training_metrics.json'
    with open(filename, 'w') as json_file:
        json.dump(training_metrics, json_file, indent=4)
    print(f"Data successfully written to {filename}")

    # # Confusion Matrices:
    # validate(model, trainloader, criterion, device, True) # Training
    # validate(model, valoader, criterion, device, True) # Validation