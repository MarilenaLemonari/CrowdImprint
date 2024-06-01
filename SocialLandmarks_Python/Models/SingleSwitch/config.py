from imports import *
from data_loader import *

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

def setup_config(wandb_bool, images, gt):
    batch_size = 64

    if wandb_bool == True:
        wandb.init(project="SocialLandmarks")
        config = wandb.config
        config.epochs = 50
        config.batch_size = batch_size

    x_train, x_val, y_train, y_val = train_test_split(images, gt, test_size=0.2, random_state=42)
    train_dataset = CustomDataset(x_train, y_train)
    val_dataset = CustomDataset(x_val, y_val)
    trainloader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valoader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return trainloader, valoader, config

if __name__ ==  '__main__':
    images, gt = load_data()
    setup_config(False, images, gt)