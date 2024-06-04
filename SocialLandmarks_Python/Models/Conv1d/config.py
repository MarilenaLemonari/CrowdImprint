from imports import *
from data_loader import *


def setup_config_conv1d(X, labels):

    batch_size = 32
    X_train, X_val, y_train, y_val = train_test_split(X, labels, test_size=0.2, random_state=42)

    # HYPERPARAMETERS
    wandb.init(project="SocialLandmarks")
    config = wandb.config
    config.epochs = 30
    config.batch_size = batch_size

    return X_train, X_val, y_train, y_val, config


if __name__ ==  '__main__':
    print("Keras or Pytorch")
    # images, gt = load_data()
    # setup_config(False, images, gt)