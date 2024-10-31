
import os
from inference_existing import *
import random
import math
import numpy as np
from generate_custom_trajectories import *
import json
from preprocess_existing import generate_python_files
from generate_trajectory_images import *

def generate_trajectories(n_gens, mode):
    n_agents = n_gens
    os.chdir("C:\PROJECTS\DataDrivenInteractionFields\InteractionFieldsUMANS\examples")

    # behavior_list = ["0_Anticlockwise_final", "1_Unidirectional_final", "2_Attractive_final", "3_Clockwise_final", "4_AvoidV2_final",
    #             "Stop", "4_AvoidOppV2_final"]
    behavior_list = ["0_Anticlockwise_final", "1_Unidirectional_final", "2_Attractive_final", "CAoval", "7_AvoidRight",
                "Stop", "6_AvoidOppp"]
    dir_dict = {0:0, 1:3}
    avoid_dict = {0:4, 1:6}

    dictionary = {}
    for i in range(len(behavior_list)-1):
        if i == 5:
            dictionary[i] = behavior_list[i+1]
        else:
            dictionary[i] = behavior_list[i]

    max_end_time = 15
    # init_positions = np.array([[0,0],[1,2],[0,0]])
    init_positions = np.zeros((n_agents*2,2))
    source_list = list(np.arange(0,n_agents) * 2) 
    build_xml(init_positions, source_list, dictionary, max_end_time)

    # combs = ["['0_5']", "['1_2']"]
    combs_dict = {}

    for r in tqdm(range(n_gens)):
        # combs_dict[f"agent_{r+1}"] = combs[r]
        # beh_combination = combs[r]
        # string = beh_combination.strip("[]'")
        # InF1, InF2 = map(int, string.split('_'))

        flag_1_4 = ''
        field_1 = random.randint(1,len(behavior_list)-2)
        field_2 = random.randint(1,len(behavior_list)-2)
        combs_dict[f"agent_{r+1}"] = str(field_1)+"_"+str(field_2)
        end_time = random.randint(6,15)
        max_radius = end_time/4 
        min_radius = 2
        radius = random.uniform(min_radius, max_radius)
        T = random.randint(3,int(end_time-3)) 

        if field_1 == 2 and field_2 == 2:
            end_time = random.randint(3,6)
            radius = end_time + 0.01
        elif field_1 == 2 and field_2 != 2:
            if T > 10:
                T = 10
            radius = T - 0.1
            if field_2 == 4:
                end_time = random.randint(8, 10)
                T = random.randint(3, int(end_time)-4)
        elif field_2 == 2:
            end_time = random.randint(6,10)
            radius = end_time/2
            if field_1 == 4:
                T = random.randint(int(end_time-4),int(end_time-2))
            else:
             T = random.randint(3,int(end_time-3))
        if field_1 == 1 and field_2 == 4:
            field_2 = 1
            field_1 = 4
            flag_1_4 = "flagged"
        if field_1 == 1 and field_2 == 1:
            end_time = 6
            radius = random.uniform(1, 5)
        if field_1 == 4:
            radius = random.uniform(3, 5)
        if field_2 == 4 and field_1 == 5:
            end_time = random.randint(8,15)
            T = random.randint(2,int(end_time-4))
        if field_1 == 5 and field_2 == 3:
            T = 2
        if field_1 == 5 and field_2 == 1:
            radius = random.uniform(1, 8)
            end_time = random.randint(4,11-int(radius))
            T = 2

        # Initialise agent and simulation duration:
        x0 = 0
        y0 = 0
        angle = random.uniform(0, 2 * math.pi)
        x = x0 + radius * math.cos(angle)
        y = y0 + radius * math.sin(angle)
        init_positions=np.array([[x0,y0],[x,y]])

        # Initialise source orientation:
        random_angle = random.uniform(0, 2 * math.pi)
        or_x = math.cos(random_angle)
        or_y = math.sin(random_angle)
        orientation = np.array([or_x, or_y])

        # Check if agent is behind source:
        vector_to_initial_agent = init_positions[1,:] - init_positions[0,:]
        dot_product = np.dot(vector_to_initial_agent, orientation)

        if field_1 == 5 and field_2 == 4:
            if dot_product <= 0:
                 # Agent behind the source:
                field_2 = 4
            else:
                # Agent in front of source:
                field_2 = 6
        elif field_1 != 5 and field_2 == 4:
            field_2 = avoid_dict[random.randint(0,1)]
            if field_1 == 1:
                field_2 = 6

        if field_1 == 4:
            if dot_product <= 0:
                # Agent behind the source:
                field_1 = 4
            else:
                # Agent in front of source:
                field_1 = 6
            if field_2 == 4 or field_2 == 6:
                field_2 = field_1


        # If sampled field is circle around randomly choose clockwise or anticlockwise options
        if field_1 == 3:
            field_dir = random.randint(0,1)
            field_1 = dir_dict[field_dir]
        if field_2 == 3:
            field_dir = random.randint(0,1)
            field_2 = dir_dict[field_dir]

        weight=np.zeros((1,len(behavior_list)-1))
        actionTimes=np.ones((1,len(behavior_list)-1))*(-1)
        inactiveTimes=np.ones((1,len(behavior_list)-1))*(-1)

        if field_1 != 5 and field_2 != 5:
            if field_1 == 6:
                first_field = 5
            else:
                first_field = field_1
            if field_2 == 6:
                sec_field = 5
            else:
                sec_field = field_2
            weight[0,first_field] = 1
            weight[0,sec_field] = 1

            inactiveTimes[0,first_field] = T
            actionTimes[0,sec_field] = T

            inactiveTimes[0,sec_field] = end_time
            actionTimes[0,first_field] = 0

        elif field_1 == 5 and field_2 != 5:
            if field_2 == 6:
                sec_field = 5
            else:
                sec_field = field_2

            weight[0,sec_field] = 1
            inactiveTimes[0,sec_field] = end_time
            actionTimes[0,sec_field] = T

        elif field_1 != 5 and field_2 == 5:
            if field_1 == 6:
                first_field = 5
            else:
                first_field = field_1

            weight[0,first_field] = 1
            inactiveTimes[0,first_field] = T
            actionTimes[0,first_field] = 0
    
        generate_instance(init_positions,weight,actionTimes,inactiveTimes,or_x, or_y,dictionary, groupID = r)
  
    # Save trajectories
    S_true=make_trajectory(n_agents,mode)

    # Save combinations:
    file_path = "C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\Trajectories" + "\\" + mode + "\comb_dict.json"
    with open(file_path, "w") as json_file:
        json.dump(combs_dict, json_file)

    return combs_dict

def get_class(combs_dict):
    class_dict = {}
    gt_dict = {"1_1": 0, "1_2": 1, "1_3": 2, "1_4": 3, "1_5": 4,
    "2_1": 5, "2_2": 6, "2_3": 7, "2_4": 8, "2_5": 9,
    "3_1": 10, "3_2": 11,"3_3": 12, "3_4": 13, "3_5": 14,
    "4_1": 15, "4_2": 16, "4_3": 17, "4_4": 18, "4_5": 19,
    "5_1": 20, "5_2": 21, "5_3": 22, "5_4": 23, "5_5": 24,
    "0_0": 12, "0_1": 10, "0_2": 11, "0_3": 12, "0_4": 13, "0_5": 14, "0_6": 13,
    "1_0": 2, "2_0": 7, "3_0": 12, "4_0": 17, "5_0": 22, "6_0":17,
    "1_6":3, "2_6":8, "3_6":13, "4_6":18, "5_6":23, "6_6":18,
    "6_1":15, "6_2":16, "6_3":17, "6_4":18, "6_5":19}

    for key,value in combs_dict.items():
        class_dict[key] = gt_dict[value]

    file_path = "C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\Trajectories" + "\\" + mode + "\class_dict.json"
    with open(file_path, "w") as json_file:
        json.dump(class_dict, json_file)

    return class_dict

def load_inference_datapoint(folder_path,npz_file):
    loaded_images = []
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

def load_python_file(dataset_name, file_name):
    # print("load_python_files()")

    folder_path = f'C:\\PROJECTS\\SocialLandmarks\\SocialLandmarks_Python\\Data\\PythonFiles\\{dataset_name}\\'  
    images = load_inference_datapoint(folder_path, file_name)

    return images

def run_model(dataset_name, combs_dict):
    
    model_name = "model_final.pth"
    model_type = "pytorch"
    # dataset_name = f"ActedScenarios/{specific}"
    predictions_dict = {}

    for key, value in combs_dict.items():
        x_test = load_python_file(dataset_name, key+".npz")
        c_batch_size = x_test.shape[0]
        batch_size = 32

        predictions, predicted_labels = model_inference(model_name, model_type, x_test, batch_size)
        predictions_dict[key] = int(predicted_labels)
    return predictions_dict

if __name__ == "__main__":
    """
    # cd C:\PROJECTS\Social_Landmarks
    # .venv\Scripts\activate
    # cd C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Experiments
    # python3 sanity_evaluation.py
    """

    mode =  "Evaluation\\Sanity"
    combs_dict = generate_trajectories(n_gens=10000, mode=mode)
    class_dict = get_class(combs_dict)
    
    generate_images(name = "\\" + mode)
    folder_path = 'C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\Images\Evaluation\Sanity'
    generate_python_files(folder_path=folder_path, name = mode)
    predictions_dict = run_model(dataset_name =  mode, combs_dict =  combs_dict)

    print(class_dict)
    print(predictions_dict)

    acc = 0
    total = 0
    for key,gt in class_dict.items():
        pred = predictions_dict[key]
        if gt == pred:
            acc += 1
        total += 1
    acc /= total
    print(acc*100, "%")