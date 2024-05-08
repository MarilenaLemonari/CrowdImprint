import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from keras.models import load_model

# Instructions:
# cd C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Experiments
# python3 inference.py

def load_trained_model(model_name, model_type):
    print("load_trained_model()")

    model_folder = "C://PROJECTS//SocialLandmarks//SocialLandmarks_Python//Models//SingleSwitch//"
    model_path = model_folder + model_name

    if model_type == "pytorch":
        trained_model = torch.load(model_path)
    elif model_type == "keras":
        trained_model = load_model(model_path)
    else:
        print("ERROR! Unsupported model type.")
    return trained_model

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

def prepare_test_data(x_test, y_test, batch_size):
    print("prepare_test_data()")

    test_dataset = CustomDataset(x_test, y_test)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return test_loader

def model_eval(model, x_test, y_test, model_type):
    print("model_eval()")

    if model_type == "pytorch":
        test_loader = prepare_test_data(x_test, y_test, batch_size)
        model.eval()
        predictions = []
        with torch.no_grad():  
            for inputs in test_loader:
                outputs = model(inputs)
                predictions.append(outputs)

        predictions = torch.cat(predictions, dim=0)
        predicted_labels = np.argmax(predictions, axis=1)
    elif model_type == "keras":
        # test_loss, test_accuracy = model.evaluate(test_loader)
        predictions = model.predict(x_test)
        predicted_labels = np.argmax(predictions, axis=1)
    else:
        print("ERROR! Unsupported model type.")

    return predictions, predicted_labels

def model_inference(model_name, model_type, x_test, y_test):
    model = load_trained_model("model_test.h5", model_type)

    predictions, predicted_labels = model_eval(model, x_test, y_test, model_type)

    return predictions, predicted_labels

if __name__ ==  '__main__':

    model_name = "model_test.h5"
    model_type = "keras"
    batch_size = 4
    x_test = np.random.random((batch_size, 32, 32, 1)) # torch.randn(batch_size, 1, 32, 32)
    y_test = np.ones(batch_size) # torch.ones(batch_size)

    predictions, predicted_labels = model_inference(model_name, model_type, x_test, y_test)
    print(predicted_labels)
    