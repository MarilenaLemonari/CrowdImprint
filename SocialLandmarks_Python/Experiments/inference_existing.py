from inference import *
import os
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from generate_custom_trajectories import *
import random
import math

# Instructions:
# cd C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Experiments
# python3 inference_existing.py

def pick_value(prob):
    return 1 if random.random() < prob else 0

def load_python_files(dataset_name):
    print("load_python_files()")

    folder_path = f'C:\\PROJECTS\\SocialLandmarks\\SocialLandmarks_Python\\Data\\PythonFiles\\{dataset_name}\\'  
    images = load_inference_data(folder_path)

    return images

def create_infered_beh_distr(predicted_labels, dataset_name, visualize = False):
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
            value, _ = decode_labels([index])
            hist_adj.append(percentage/100)
            tick_names.append(str())
            beh_distr[str(value)] = str(percentage) + "%"
            # print(percentage, "% ",value,"behavior.")

    file_path = f"Inference/{dataset_name}_inferred_behavior_distribution.json"
    with open(file_path, "w") as json_file:
        json.dump(beh_distr, json_file)

    if visualize:
        plt.bar(range(len(beh_distr)-1), hist_adj, tick_label=tick_names, color = 'slategrey', edgecolor = 'dimgrey')
        plt.xlabel('Inferred Behaviour Combination')
        plt.ylabel('Predicted Behaviour Frequency (%)')
        plt.title(f'Inferred Behaviour Distribution of {dataset_name} Dataset')
        plt.savefig(f"Inference/{dataset_name}_inferred_behavior_distribution.png")
        plt.show()

    return beh_distr

def generate_trajectories(beh_distr, n_agents, mode):
    print("generate_trajectories()")

    # TODO: do the percentage mean probabilities? 
    gen_beh = {}
    combs = []
    while len(combs) < n_agents:
        for key, value in beh_distr.items():
            if key == "dataset":
                continue
            if len(combs) >= n_agents:
                continue
            float_value = float(value.split('%')[0])
            prob = float_value / 100
            value = pick_value(prob)
            if value == 1:
                combs.append(key)
                if key in gen_beh:
                    gen_beh[key] += 1
                else:
                    gen_beh[key] = 1

    # If not:
    # Count how many agents will share which behavior combinations:
    # gen_beh = {}
    # combs = []
    # for key, value in beh_distr.items():
    #     if key == "dataset":
    #         continue
    #     float_value = float(value.split('%')[0])
    #     n_values = int(np.round((float_value * n_agents) / 100))
    #     if n_values != 0:
    #         gen_beh[key] = n_values
    #         for v in range(n_values):
    #             combs.append(key)   
    # n_agents = len(combs)

    os.chdir("C:\PROJECTS\DataDrivenInteractionFields\InteractionFieldsUMANS\examples")
    behavior_list = ["0_Anticlockwise_final", "1_Unidirectional_final", "2_Attractive_final", "3_Clockwise_final", "4_AvoidV2_final",
                "Stop", "4_AvoidOppV2_final"]

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

    for r in tqdm(range(len(combs))):
        combs_dict[f"agent_{r+1}"] = combs[r]
        beh_combination = combs[r]
        string = beh_combination.strip("[]'")
        InF1, InF2 = map(int, string.split('_'))
        end_time = random.randint(6,max_end_time)
        max_radius = end_time/4 
        min_radius = 0.5
        radius = random.uniform(min_radius, max_radius)
        T = random.randint(3,int(end_time-3))
        
        # Sample Circle Around Behaviour:
        if InF1 == 0 or InF1 == 3:
            flag_ca = random.randint(0,1)
            if flag_ca == 0:
                InF1 = 0
            else:
                InF1 = 3
        if InF2 == 0 or InF2 == 3:
            flag_ca = random.randint(0,1)
            if flag_ca == 0:
                InF2 = 0
            else:
                InF2 = 3

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

        # Handle Avoid Behaviour:
        if InF1 == 4 or InF1 == 6:
            # check if behind:
            vector_to_initial_agent = init_positions[1,:] - init_positions[0,:] 
            dot_product = np.dot(vector_to_initial_agent, orientation)
            if dot_product <= 0:
                # Agent behind the source:
                InF1 = 4
            else:
                # Agent in front of source:
                InF1 = 6
        if InF2 == 4 or InF2 == 6:
            flag_av = random.randint(0,1)
            if flag_av == 0:
                InF2 = 4
            else:
                InF2 = 6
        field_1 = InF1
        field_2 = InF2

        weight = np.zeros((1,len(behavior_list)-1))
        actionTimes = np.ones((1,len(behavior_list)-1))*(-1)
        inactiveTimes = np.ones((1,len(behavior_list)-1))*(-1)

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


if __name__ ==  '__main__':

    model_name = "trial2.pth"
    model_type = "pytorch"
    # dataset_name = "Flock"
    # dataset_name = "Zara"
    dataset_name = "Students"

    x_test = load_python_files(dataset_name)
    c_batch_size = x_test.shape[0]
    batch_size = 32 # TODO: check
    # if c_batch_size >= train_batch_size:
    #     batch_size = train_batch_size
    #     print("WARNING! Need Test Loader.")
    #     testloader= prepare_test_data(x_test, batch_size)
    # else:
    #     batch_size = c_batch_size
    # x_test = np.random.random((batch_size, 32, 32, 1)) # torch.randn(batch_size, 1, 32, 32)
    # y_test = np.ones(batch_size) # torch.ones(batch_size)

    predictions, predicted_labels = model_inference(model_name, model_type, x_test, batch_size)
    combinations, c_dict = decode_labels(predicted_labels)
    print(c_dict)

    beh_distr = create_infered_beh_distr(predicted_labels, dataset_name, True)
    print(beh_distr)
    
    # Generate new trajectories based on inferred behaviours:
    n_agents = 3
    mode =  f"Inference\{dataset_name}"
    generate_trajectories(beh_distr, n_agents, mode)