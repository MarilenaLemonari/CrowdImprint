from inference import *
from inference_existing import *
from preprocess_existing import *
from generate_custom_trajectories import *
import os
from tqdm import tqdm

# Instructions:
# cd C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Experiments
# python3 evaluation_existing.py

def generate_guided_trajectory(beh_combination, start_pts, or_pts, source_pt, end_time, timestep, name):
    start_x, start_z = start_pts
    or_x = or_pts[0] - start_x
    or_z = or_pts[1] - start_z

    os.chdir("C:\PROJECTS\DataDrivenInteractionFields\InteractionFieldsUMANS\examples")
    behavior_list = ["Unidirectional_Down","Attractive_Multidirectional","Other_CircleAround", "AvoidNew", "MoveTF", "Stop"]

    dictionary = {}
    for i in range(len(behavior_list)-1):
        dictionary[i] = behavior_list[i]

    
    init_positions = np.array([[source_pt,source_pt],[start_x,start_z]])
    build_xml(init_positions, [0], dictionary, end_time, delta_time = timestep)


    string = beh_combination.strip("[]'")
    InF1, InF2 = map(int, string.split('_'))
    field_1 = InF1
    field_2 = InF2

    weight = np.zeros((1,len(behavior_list)-1))
    actionTimes = np.ones((1,len(behavior_list)-1))*(-1)
    inactiveTimes = np.ones((1,len(behavior_list)-1))*(-1)
    T = random.randint(2,int(end_time-2)) # TODO: Find most optimal?

    if field_1 != 5 and field_2 != 5:
        weight[0,field_1] = 1
        weight[0,field_2] = 1

        inactiveTimes[0,field_1] = T
        actionTimes[0,field_2] = T

        inactiveTimes[0,field_2] = end_time
        actionTimes[0,field_1] = 0
    elif field_1 == 5 and field_2 != 5:
        weight[0,field_2] = 1
        inactiveTimes[0,field_2] = end_time
        actionTimes[0,field_2] = T
    elif field_1 != 5 and field_2 == 5:
        weight[0,field_1] = 1
        inactiveTimes[0,field_1] = T
        actionTimes[0,field_1] = 0

    generate_instance(init_positions,weight,actionTimes,inactiveTimes,or_x, or_z,dictionary, groupID = 0)
    
    # Save trajectories
    mode = "Evaluation\\Flock"
    S_true=make_trajectory(1,mode,name = name)

    return S_true


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

    # beh_distr = create_infered_beh_distr(predicted_labels, dataset_name = "Flock", visualize = False)
    # print(beh_distr)

    # Step ii: Trajectory generation
    current_file_dir = "C:\PROJECTS\SocialLandmarks\Data\Trajectories"
    name = "\Flock"
    csv_directory  = current_file_dir + name + "\\"
    csv_data = read_csv_files(csv_directory)
    timestep = 0.04
    for i, agent_csv in enumerate(csv_data):
        start_x = csv_data[agent_csv]["pos_x"][0]
        start_z = csv_data[agent_csv]["pos_z"][0]
        # plt.plot(csv_data[agent_csv]["pos_x"], csv_data[agent_csv]["pos_z"])
        # plt.show()
        # exit()
        num_steps = len(csv_data[agent_csv]["pos_x"])
        end_time = (num_steps * timestep) - timestep
        start_pts = [start_x, start_z]
        or_pts = [csv_data[agent_csv]["pos_x"][2], csv_data[agent_csv]["pos_z"][2]]
        source_pt = csv_data[agent_csv]["norm_source"][0]
        prod_traj = generate_guided_trajectory(combinations[i], start_pts, or_pts, source_pt, end_time, timestep, agent_csv)
        eval_x = prod_traj[0][:,1]
        eval_z = prod_traj[0][:,2]
        plt.plot(eval_x, eval_z)
        plt.show()
        exit()
    

