from imports import *
from data_loader import *


def setup_config_keras(x_train_a, x_val_a, x_train_b, x_val_b, y_train, y_val):

    batch_size = 32
    #x_train_a, x_val_a, x_train_b, x_val_b, y_train, y_val = train_test_split(images_a, images_b, gt, test_size=0.2, random_state=42)
    x_train_a = np.expand_dims(x_train_a, axis=-1)
    x_train_b = np.expand_dims(x_train_b, axis=-1)
    x_val_a = np.expand_dims(x_val_a, axis=-1)
    x_val_b = np.expand_dims(x_val_b, axis=-1)
    y_train = np.expand_dims(y_train, axis=-1)
    y_val = np.expand_dims(y_val, axis=-1)
    # x_train_a = np.transpose(x_train_a, (1, 2, 0))
    # x_train_b = np.transpose(x_train_b, (1, 2, 0))
    # x_val_a = np.transpose(x_val_a, (1, 2, 0))
    # x_val_b = np.transpose(x_val_b, (1, 2, 0))

    # HYPERPARAMETERS
    # wandb.init(project="SocialLandmarks")
    # config = wandb.config
    epochs = 1 #TODO
    batch_size = batch_size

    return x_train_a, x_train_b, y_train, x_val_a, x_val_b,  y_val, epochs, batch_size


if __name__ ==  '__main__':
    print("Keras or Pytorch")
    # images, gt = load_data()
    # setup_config(False, images, gt)