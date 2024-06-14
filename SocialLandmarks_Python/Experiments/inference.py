from imports import *
from CNNPytorch import *

# Instructions:
# cd C:\PROJECTS\SocialLandmarks
# Execute .\.venv\Scripts\activate
# cd C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Experiments
# python3 inference.py

def load_inference_data(folder_path):

    file_list = os.listdir(folder_path)
    npz_files = [file for file in file_list if file.endswith('.npz')]
    loaded_images = []

    for npz_file in tqdm(npz_files): 
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

    images = np.array(loaded_images)
    images = images[:, np.newaxis, :, :]

    return images

def load_trained_model(model_name, model_type):
    print("load_trained_model()")

    model_folder = "C://PROJECTS//SocialLandmarks//SocialLandmarks_Python//Models//SingleSwitch//"
    model_path = model_folder + model_name

    if model_type == "pytorch":
        trained_model, criterion, optimizer, device = instantiate_model()
        trained_model.load_state_dict(torch.load(f"{model_path}"))
    elif model_type == "keras":
        trained_model = load_model(model_path)
    else:
        print("ERROR! Unsupported model type.")
    return trained_model

class CustomDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        image = torch.from_numpy(image)

        return image

def prepare_test_data(x_test, batch_size):
    print("prepare_test_data()")

    test_dataset = CustomDataset(x_test)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader

def model_eval(model, x_test, model_type, batch_size):
    print("model_eval()")

    if model_type == "pytorch":
        test_loader = prepare_test_data(x_test, batch_size)
        model.eval()
        predictions = []
        with torch.no_grad():  
            for i, (inputs) in enumerate(test_loader):
                outputs = model(inputs.float())
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

def model_inference(model_name, model_type, x_test, batch_size):
    model = load_trained_model(f"{model_name}", model_type)

    predictions, predicted_labels = model_eval(model, x_test, model_type, batch_size)

    return predictions, predicted_labels

def decode_labels(predicted_labels):

    decoder = {0:"1_1", 1:"1_2",2: "1_3", 3: "1_4", 4: "1_5",
            5: "2_1", 6: "2_2", 7: "2_3", 8: "2_4", 9: "2_5",
            10 :"3_1", 11: "3_2",12: "3_3", 13: "3_4", 14: "3_5",
            15: "4_1", 16: "4_2", 17: "4_3", 18: "4_4", 19: "4_5",
            20: "5_1", 21: "5_2", 22: "5_3", 23: "5_4", 24: "5_5"}
    
    combinations = []
    for i in (range(len(predicted_labels))):
        combo = decoder[int(predicted_labels[i])]
        combinations.append(combo)
    return combinations

if __name__ ==  '__main__':

    model_name = "trial2.pth"
    model_type = "pytorch"
    batch_size = 32
    folder_path = "C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\PythonFiles\SingleSwitch\TestData"
    # x_test = np.random.random((batch_size, 1, 32, 32)) 
    # y_test = np.ones(batch_size) 
    x_test = load_inference_data(folder_path)

    predictions, predicted_labels = model_inference(model_name, model_type, x_test)
    print(predicted_labels)

    combinations = decode_labels(predicted_labels)
    print(combinations)
    