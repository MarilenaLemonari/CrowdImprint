from inference import *
import os
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

# Instructions:
# cd C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Experiments
# python3 inference_existing.py

# HELPER FUNCTIONS:
def scale_to_standard_normal(images):
    mean = np.mean(images)
    std = np.std(images)
    scaled_images = (images - mean) / std
    return scaled_images
# #

def load_python_files(dataset_name):
    print("load_python_files()")

    folder_path = f'C:\\PROJECTS\\SocialLandmarks\\SocialLandmarks_Python\\Data\\PythonFiles\\{dataset_name}\\'  
    file_list = os.listdir(folder_path)
    npz_files = [file for file in file_list if file.endswith('.npz')]
    loaded_images = []

    for npz_file in tqdm(npz_files):
        file_path = os.path.join(folder_path, npz_file)
        loaded_data = np.load(file_path)
        array_keys = loaded_data.files
        array_key = array_keys[0]
        array = loaded_data[array_key]
        if (array.dtype != 'float32'):
            print(file_path)
            exit()
        loaded_images.append(array)

    x = scale_to_standard_normal(loaded_images)
    return x

def decode_predictions(prediction_labels):
    # print("decode_predictions()")

    dec_dict = {0: "0_0", 1: "0_1", 2: "0_2", 3: "0_3", 4: "0_4",5: "0_5",
        6: "1_0",7: "1_1",8: "1_2",9: "1_3", 10: "1_4", 11: "1_5", 
        12: "2_0", 13: "2_1", 14: "2_2", 15:"2_3", 16: "2_4", 17: "2_5",
            18: "3_0", 19: "3_1", 20: "3_2", 21: "3_3", 22: "3_4", 23: "3_5",
            24: "4_0", 25: "4_1", 26: "4_2", 27: "4_3", 28: "4_4", 29: "4_5",
            30: "5_0", 31: "5_1", 32: "5_2", 33: "5_3", 34: "5_4", 35: "5_5"}
    
    combinations  = []
    for pred in prediction_labels:
        combinations.append(dec_dict[int(pred)])

    return combinations

def create_infered_beh_distr(predicted_labels, visualize = False):
    hist, bins = np.histogram(predicted_labels, bins=np.arange(max(predicted_labels) + 2))
    count_list = np.array(hist.tolist())
    distribution = (count_list * 100) / sum(count_list)
    beh_distr = {"dataset": f"{dataset_name}"}
    tick_names = []
    hist_adj = []
    for index, percentage in enumerate(distribution):
        if percentage == 0:
            continue
        else:
            hist_adj.append(percentage/100)
            tick_names.append(str(decode_predictions([index])))
            beh_distr[str(decode_predictions([index]))] = str(percentage) + "%"
            # print(percentage, "% ",decode_predictions([index]),"behavior.")

    file_path = f"Inference/{dataset_name}_inferred_behavior_distribution.json"
    with open(file_path, "w") as json_file:
        json.dump(beh_distr, json_file)

    if visualize:
        plt.bar(range(len(beh_distr)-1), hist_adj, tick_label=tick_names, color = 'slategrey')
        plt.xlabel('Inferred Behaviour Combination')
        plt.ylabel('Predicted Behaviour Frequency (%)')
        plt.title(f'Inferred Behaviour Distribution of {dataset_name} Dataset')
        plt.savefig(f"Inference/{dataset_name}_inferred_behavior_distribution.png")
        plt.show()

    return beh_distr

def generate_trajectories(beh_distr):
    print("generate_trajectories()")

if __name__ ==  '__main__':

    model_name = "model_test.h5"
    model_type = "keras"
    dataset_name = "Flock"

    x_test = load_python_files(dataset_name)
    c_batch_size = x_test.shape[0]
    train_batch_size = 32 # TODO: check
    if c_batch_size >= train_batch_size:
        batch_size = train_batch_size
        print("WARNING! Need Test Loader.")
        exit()
    else:
        batch_size = c_batch_size
    # x_test = np.random.random((batch_size, 32, 32, 1)) # torch.randn(batch_size, 1, 32, 32)
    y_test = np.ones(batch_size) # torch.ones(batch_size)

    predictions, predicted_labels = model_inference(model_name, model_type, x_test, y_test)
    combinations = decode_predictions(predicted_labels)

    beh_distr = create_infered_beh_distr(predicted_labels, True)
    
    # Generate new trajectories based on inferred behaviours:
    generate_trajectories(beh_distr)

    