from imports import *
from data_loader import *

# python3 C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\SingleSwitch\config.py

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

def setup_config_keras():
    #dataset = CustomDataset(images,gt)

    batch_size = 32
    #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # x_train, x_val, y_train, y_val = train_test_split(images, gt, test_size=0.2, random_state=42)

    # HYPERPARAMETERS
    wandb.init(project="SocialLandmarks")
    config = wandb.config
    config.epochs = 30
    config.batch_size = batch_size

    datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
    )
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    return config, datagen, reduce_lr

def setup_config(wandb_bool, images, gt):  

    batch_size = 32

    if wandb_bool == True:
        wandb.init(project="SocialLandmarks")
        config = wandb.config
        config.epochs = 20
        config.batch_size = batch_size
    else:
        config = []

    x_train, x_val, y_train, y_val = train_test_split(images, gt, test_size=0.2, random_state=42)
    print(x_train.shape, x_val.shape, len(y_train), len(y_val))
    train_dataset = CustomDataset(x_train, y_train)
    val_dataset = CustomDataset(x_val, y_val)
    trainloader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valoader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return trainloader, valoader, config

def setup_config_test(images, gt):
    batch_size = 32
    dataset = CustomDataset(images, gt)
    dataloader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


if __name__ ==  '__main__':
    print("Keras or Pytorch")
    images, gt = load_data()
    setup_config(False, images, gt)