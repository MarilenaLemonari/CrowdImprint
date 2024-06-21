from inference import *
from inference_existing import *
from preprocess_existing import *
from generate_custom_trajectories import *
import os
from tqdm import tqdm

# Instructions:
# cd C:\PROJECTS\SocialLandmarks
# .\.venv\Scripts\activate
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


    """
    A measure of the "stopping" behaviour.
    This corresponds to the amount of time that speed is 0 (or less than a tolerance).
    """
    return trajectory

def read_value_csv_files(csv_directory):
    csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]

    data_dict = {}
    all_dfs = []
    column_names = ['timestep','pos_x', 'pos_z']

    row_threshold = 3
    for filename in tqdm(csv_files):
        # Read the CSV file into a pandas DataFrame and assign column names
        df = pd.read_csv(os.path.join(csv_directory, filename), 
            header=None, names=column_names, 
            skiprows=None,
            #skiprows=lambda index: index == 0 or skip_rows(index, row_step),
            usecols=[0, 1, 2])
        df[column_names[0]] = df[column_names[0]].astype(float) 
        df[column_names[1]] = df[column_names[1]].astype(float)
        df[column_names[2]] = df[column_names[2]].astype(float)
        # df.drop(0, axis=1, inplace=True)
        if df.shape[0] < row_threshold:
            continue

        df["speed"] = 0
        for i in range(1, len(df)):
            df.loc[i, "speed"] = math.sqrt((df.loc[i, 'pos_x'] - df.loc[i - 1, 'pos_x']) ** 2 + (df.loc[i, 'pos_z'] - df.loc[i - 1, 'pos_z']) ** 2)

        df.drop("timestep", axis=1, inplace=True)
        data_dict[filename] = df
        all_dfs.append(df)
    
    for filename, df in data_dict.items():
        # Normalize to [0, 1]
        bound_min = min(np.min(df["pos_x"]), np.min(df["pos_z"]), 0)
        bound_max = max(np.max(df["pos_x"]), np.max(df["pos_z"]), 0)

        bound_max += 0.5
        bound_min -= 0.5 

        df["pos_x"] = (df['pos_x'] - bound_min) / (bound_max - bound_min) * (1 - 0) 
        df["pos_z"] = (df['pos_z'] - bound_min) / (bound_max - bound_min) * (1 - 0) 


        s = len(df["pos_x"])
        source_norm = np.zeros((s))
        source_norm[0] = (0 - bound_min) / (bound_max - bound_min) * (1 - 0)
        df["norm_source"] = list(source_norm)

        data_dict[filename] = df

    return data_dict

def centre_and_rotate(agent_traj):
        # plt.plot(agent_traj["pos_x"].values, agent_traj["pos_z"].values, 'slategrey')
        # plt.title("Raw Trajectories")
        # plt.show()

        # Centre starting point at origin (0,0):
        start_pos_x = agent_traj["pos_x"].iloc[0]
        start_pos_y = agent_traj["pos_z"].iloc[0]
        x_s = agent_traj["pos_x"]-start_pos_x
        y_s = agent_traj["pos_z"]-start_pos_y
        # plt.plot(x_s, y_s)
        # plt.show()

        # Rotate to align end points:
        end_pos_x = x_s.iloc[-1]
        end_pos_y = y_s.iloc[-1]
        if end_pos_y != 0:
            tan_value = end_pos_x/end_pos_y
            theta = abs(np.arctan(tan_value))
            """
            if end_pos_x > 0 and end_pos_y > 0:
                theta = (np.deg2rad(90) - theta)
            elif end_pos_x > 0 and end_pos_y <= 0:
                theta = - (np.deg2rad(90) - theta)
            elif end_pos_x <= 0 and end_pos_y > 0:
                theta =  np.deg2rad(90) + theta
            else:
                theta = - (np.deg2rad(90) + theta)
            """
            if end_pos_x > 0 and end_pos_y > 0:
                theta = (np.deg2rad(180) - theta)
            elif end_pos_x > 0 and end_pos_y <= 0:
                theta = theta
            elif end_pos_x <= 0 and end_pos_y > 0:
                theta = - (np.deg2rad(180) - theta)
            else:
                theta = - theta
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            points = np.column_stack((x_s, y_s))
            rotated_points = np.dot(points, rotation_matrix)
            # plt.plot(rotated_points[:,0], rotated_points[:,1], 'slategrey')
            # plt.show()
            max_dist_value = rotated_points[-1,1]
            max_width_value = max(rotated_points[:,0])
            
        else:
            rotated_points = np.array([[0,0]])
            max_dist_value = 0
            max_width_value = 0
        return rotated_points, max_dist_value, max_width_value

def stretch_points(points, max_value):
    if points[-1,1] == 0:
        y_stretched = points[:,1]
        # plt.plot(points[:,0], y_stretched, '*')
    else:
        scale_factor = max_value/points[-1,1]
        y_stretched = [scale_factor * yi for yi in points[:,1]]
        # plt.plot(points[:,0], y_stretched, 'slategrey')
    return y_stretched

def prepare_search_database(values_dict):
    processed_points = []
    max_dist = []
    max_width =[]
    database_dict = {}
    counter = 0
    for search_key, agent_traj in values_dict.items():

        rotated_points, max_dist_value, max_width_value = centre_and_rotate(agent_traj)
        processed_points.append(rotated_points)
        max_dist.append(max_dist_value)
        max_width.append(max_width_value)
        
        database_dict[counter] = search_key
        counter += 1

    # Stretch points:
    max_value = max_dist[np.argmax(np.abs(max_dist))]
    max_height = []
    data = []
    for i, points in enumerate(processed_points):   
        y_stretched = stretch_points(points, max_value)
        max_height.append(max(np.abs(y_stretched)))
        datapoints = np.column_stack((points[:,0],np.array(y_stretched)))
        data.append(datapoints)
        values_dict[database_dict[i]] = datapoints
    # plt.show()

    return data, values_dict, max_value


if __name__ ==  '__main__':
    """
    Evaluation with existing dataset is limited to measuring % success in behavior replication.
    The following code then does the following:
        i. 
        iii. Trajectory comparison with inputs/gts given same initial positions.
        WE assume appropriate source placement.
    """

    model_name = "trial2.pth"
    model_type = "pytorch"
    dataset_name = "Flock"
    # dataset_name = "Zara"
    # dataset_name = "Students"
    folder_path = f'C:\\PROJECTS\\SocialLandmarks\\SocialLandmarks_Python\\Data\\PythonFiles\\{dataset_name}\\'  
    x_test, pred_dict = load_inference_data(folder_path, return_dict=True)
    c_batch_size = x_test.shape[0]
    batch_size = 32
    predictions, predicted_labels = model_inference(model_name, model_type, x_test, batch_size)
    combinations, c_dict = decode_labels(predicted_labels, pred_dict)
    # print(c_dict)

    # Load trajectories:
    query_file_dir = "C:\PROJECTS\SocialLandmarks\Data\Trajectories"
    query_name = "\Flock"
    query_traj_directory  = query_file_dir + query_name + "\\"
    values_traj_directory  = "C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\Trajectories\SingleSwitch"
    queries_dict = read_csv_files(query_traj_directory)
    values_dict = read_value_csv_files(values_traj_directory)    # print(values_dict['0IF_0_1_T5_d8_a1.csv'])

    # Prepare search database:
    data, values_dict, max_value = prepare_search_database(values_dict)        
    
    # Search database with query:
    search_dict = {}
    for query, query_traj in tqdm(queries_dict.items()):
        # Prepare query for searching:
        rotated_points, max_dist_value, max_width_value = centre_and_rotate(query_traj)
        y_stretched = stretch_points(rotated_points, max_value)

        dist_dict = {}
        for search_key, search_value in values_dict.items():
            """
            Search values have timestep 0.1s + they start from 0.1s.
            Query values have timesteps according to their source e.g., Flock = 0.04s, + they start for 0s.
            """
            curve1 = np.column_stack((rotated_points[:,0], y_stretched))
            curve2 = search_value           
            distance, path = fastdtw(curve1, curve2, dist=euclidean)
            dist_dict[search_key] = distance
        found_key = min(dist_dict, key=dist_dict.get)
        query_new = query.split('.csv')[0]
        search_dict[query_new] = found_key
    # print(search_dict)

    final_dict = {}
    metric = 0
    # TODO: confusion matrix
    for key in c_dict.keys():
        key_new = key.split('.npz')[0]
        pred = c_dict[key]
        pseudo = search_dict[key_new]
        pseudo_gt = pseudo.split('IF_')[1].split('_T')[0]
        final_dict[key_new] = [pred,pseudo_gt]
        if pred == pseudo_gt:
            metric += 1
    print(final_dict)
    print("Accuracy: ", (metric/len(c_dict.keys())) * 100, " %")