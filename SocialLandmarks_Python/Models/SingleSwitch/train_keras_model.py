from imports import *
from data_loader_keras import *
from CNNKeras import *

os.environ['WANDB_API_KEY']="29162836c3095b286c169bf889de71652ed3201b"

# TODO:
# cd C:\PROJECTS\SocialLandmarks
# Execute .\.venv\Scripts\activate 
# go to cd C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data
# Execute python3 C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\SingleSwitch\train_keras_model.py

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
    
# if __name__ ==  '__main__':
#     """
#     Train Pytorch model to predict combinations of InF.
#     Preprocessing:
#         Load the npy files and build gt (x,y) pairs.
#     Inputs:
#         Model inputs are (1,32,32) 
#     Outputs:
#         Model outputs are predictions of 1st and 2nd InF.
#         The code saves the trained model,
#         and the confusion matrices. 
#     """

images, gt  = load_data()
print("SUCCESS! Data Loaded. Details: ", images.shape, len(gt))

dataset = CustomDataset(images,gt)
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(images, gt, test_size=0.2, random_state=42)

model = instantiate_model()

# HYPERPARAMETERS
wandb.init(project="SocialLandmarks")
config = wandb.config
config.epochs = 20
config.batch_size = batch_size

model.fit(x_train, y_train, epochs=config.epochs, batch_size=config.batch_size,
        validation_data=(x_val, y_val),
        callbacks=[WandbCallback()])
# model.save("C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\SingleSwitch\model_test.h5")
# model.save("C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\SingleSwitch\model_test.pth")
print("MODEL IS SAVED!!")
wandb.finish()

# CONFUSION MATRICES:

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