from inference import *
from inference_existing import *
from preprocess_existing import *
import os
from tqdm import tqdm

# Instructions:
# cd C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Experiments
# python3 evaluation_existing.py


if __name__ ==  '__main__':
    """
    Evaluation with existing dataset is limited to measuring % success in behavior replication.
    The following code then does the following:
        i. Inference on existing dataset trajectories.
        ii. New trajectory generation based on infered behaviors.
        iii. Trajectory comparison with inputs/gts given same initial positions.
        WE ASSUMME appropriate source placement.
    """

    # Step i: Inference
    model_name = "model_test.h5"
    model_type = "keras"
    x_test = load_python_files("Flock")
    c_batch_size = x_test.shape[0]
    train_batch_size = 32 # TODO: check
    if c_batch_size >= train_batch_size:
        batch_size = train_batch_size
        print("WARNING! Need Test Loader.")
        exit()
    else:
        batch_size = c_batch_size
    y_test = np.ones(batch_size) # torch.ones(batch_size)
    predictions, predicted_labels = model_inference(model_name, model_type, x_test, y_test)
    combinations = decode_predictions(predicted_labels)

    # Step ii: Trajectory generation
    current_file_dir = "C:\PROJECTS\SocialLandmarks\Data\Trajectories"
    name = "\Flock"
    csv_directory  = current_file_dir + name + "\\"
    csv_data = read_csv_files(csv_directory)
    start_x = csv_data["agent_0.csv"]["pos_x"][0]
    start_z = csv_data["agent_0.csv"]["pos_z"][0]
    print(start_x, start_z)
    exit()
    # n_csvs = len(csv_data)
    # dict_list = list(csv_data.items())
    # for i in tqdm(range(n_csvs)):
    #     key, value = dict_list[i]

    behavior_dict = {0:"Unidirectional", 1:"Attractive_Multidirectional", 2:"CircleAround", 
                     3: "Avoid", 4: "MoveTF", 5: "Stop"}
    behavior_list = ["Unidirectional_Down","Attractive_Multidirectional","Other_CircleAround", "AvoidNew", "MoveTF", "Stop"]
    agent_0_behavior = combinations[0]
    print(agent_0_behavior)
