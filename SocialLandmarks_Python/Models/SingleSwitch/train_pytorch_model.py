# IMPORTS
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torch.nn.functional as F

# python3 C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\SingleSwitch\train_pytorch_model.py

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
gt_dict = {"1_1": 0, "1_2": 1, "1_3": 2, "1_4": 3, "1_5": 4, "1_6": 5,
        "2_1": 6, "2_2": 7, "2_3": 8, "2_4": 9, "2_5": 10, "2_6": 11,
        "3_1": 12, "3_2": 13,"3_3": 14, "3_4": 15, "3_5": 16, "3_6": 17,
        "4_1": 18, "4_2": 19, "4_3": 20, "4_4": 21, "4_5": 22, "4_6":23,
        "5_1": 24, "5_2": 25, "5_3": 26, "5_4": 27, "5_5": 28, "5_6": 29,
        "6_1": 30, "6_2": 31, "6_3": 32, "6_4": 33, "6_5": 34, "6_6": 35,
        "0_0": 14, "0_1": 12, "0_2": 13, "0_3": 14, "0_4": 15, "0_5": 16, "0_6": 17,
        "1_0": 2, "2_0": 8, "3_0": 14, "4_0": 20, "5_0": 26, "6_0": 32}
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
# TODO: 2 predictions of 6.
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
        image = torch.from_numpy(image).float()
        return image.unsqueeze(0), label

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.1)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout2 = nn.Dropout(p=0.1)
        
        self.fc1 = nn.Linear(32 * 8 * 8, 2048)
        self.bn3 = nn.BatchNorm1d(2048)
        self.dropout3 = nn.Dropout(p=0.2)
        
        self.fc2 = nn.Linear(2048, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.dropout4 = nn.Dropout(p=0.2)
        
        self.fc3 = nn.Linear(1024, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.dropout5 = nn.Dropout(p=0.2)
        
        self.fc4 = nn.Linear(128, 36)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        
        x = x.view(-1, 32 * 8 * 8)
        
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout3(x)
        
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.dropout4(x)
        
        x = F.relu(self.bn5(self.fc3(x)))
        x = self.dropout5(x)
        
        x = self.fc4(x)
        return x

# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 100

# Dataset and DataLoader
dataset = CustomDataset(x, gt)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model, Loss, and Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.long().to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_loss = epoch_loss / len(train_loader)
    train_acc = correct / total
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Train Accuracy: {train_acc}')
    
    # Validation Loop
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.long().to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = val_loss / len(val_loader)
    val_acc = correct / total
    print(f'Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss}, Val Accuracy: {val_acc}')

# Save Model
torch.save(model.state_dict(), 'model_test.pth')
print("MODEL IS SAVED!!")

# Confusion Matrix
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

confusion = confusion_matrix(y_true, y_pred)
print("Confusion Matrix for Validation Data:")
print(confusion)

# Visualization of Confusion Matrix
plt.figure(figsize=(10, 8))
plt.imshow(confusion, cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
