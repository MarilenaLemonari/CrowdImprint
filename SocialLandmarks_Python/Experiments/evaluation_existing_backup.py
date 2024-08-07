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
# python3 evaluation_existing_backup.py

def read_value_csv_files(csv_directory):
    csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]

    data_dict = {}
    all_dfs = []
    column_names = ['timestep','pos_x', 'pos_z']

    row_threshold = 3
    for filename in tqdm(csv_files): # TODO
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

def evaluate_trajectories():
    # METHOD 1: Compare trajectories

    # Load trajectories:
    query_file_dir = "C:\PROJECTS\SocialLandmarks\Data\Trajectories"
    # query_name = "\Flock"
    # query_traj_directory  = query_file_dir + query_name + "\\"
    query_traj_directory  =  "C:\PROJECTS\SocialLandmarks\Data\Tracking\YOLO\Tracker\Trajectories"
    values_traj_directory  = "C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\Trajectories\SingleSwitch"
    queries_dict = read_value_csv_files(query_traj_directory) #TODO: was read_csv_files(query_traj_directory)
    values_dict = read_value_csv_files(values_traj_directory)    # print(values_dict['0IF_0_1_T5_d8_a1.csv'])

    # Prepare search database:
    data_v, values_dict, max_value = prepare_search_database(values_dict)   
    
    # Search database with query:
    search_dict = {}
    json_path = "C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Experiments\Evaluation"
    json_name = json_path + "\search_dict_traj.json"
    for query, query_traj in queries_dict.items():
        # Prepare query for searching:
        rotated_points, max_dist_value, max_width_value = centre_and_rotate(query_traj)
        y_stretched = stretch_points(rotated_points, max_value)

        dist_dict = {}
        for search_key, search_value in tqdm(values_dict.items()):
            curve1 = np.column_stack((rotated_points[:,0], y_stretched))
            curve2 = search_value           
            distance, path = fastdtw(curve1, curve2, dist=euclidean)
            dist_dict[search_key] = distance
        found_key = min(dist_dict, key=dist_dict.get)
        query_new = query.split('.csv')[0]
        search_dict[query_new] = found_key
        with open(json_name, 'w') as json_file:
            json.dump(search_dict, json_file, indent=4)

    return search_dict, json_path

def evaluate_images(query_path):
    # METHOD 2: Compare Images
    # Load images:
    search_path = 'C:/PROJECTS/SocialLandmarks/SocialLandmarks_Python/Data/Images/SingleSwitch_FINAL' # TODO
    all_search_pnts = os.listdir(search_path)
    tif_s = [file for file in all_search_pnts if file.lower().endswith('.tif')]

    # query_path = 'C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\Images\Flock'
    all_query_pnts = os.listdir(query_path)
    tif_q = [file for file in all_query_pnts if file.lower().endswith('.tif')]
    
    search_dict = {}
    json_path = "C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Experiments\Evaluation"
    json_name = json_path + "\search_dict.json"
    for query in tqdm(tif_q):
        image_path = os.path.join(query_path, query)
        image1 = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        # cv2.imshow('image',image)
        # cv2.waitKey(0)
        # cv2.imwrite(os.curdir + "imag.tif", image)
        query_new = query.split('.tif')[0]
        dist_dict = {}
        for search_value in tif_s: 
            image_path = os.path.join(search_path, search_value)
            image2 = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            (score, diff) = ssim(image1, image2, full=True, data_range=image1.max() - image1.min())
            dist_dict[search_value] = score
        found_key = max(dist_dict, key=dist_dict.get)
        search_dict[query_new] = found_key
        with open(json_name, 'w') as json_file:
            json.dump(search_dict, json_file, indent=4)

    return search_dict, json_path

def evaluate_dedicated_metric(traj_directory):
    # METHOD 3:
    print("evaluate_dedicated_metrics()")

    # Load trajectories:
    # file_dir = "C:\PROJECTS\SocialLandmarks\Data\Trajectories"
    # name = "\Flock"
    # traj_directory  = file_dir + name + "\\"
    traj_dict = read_value_csv_files(traj_directory)
    
    final_dict = {}
    for agent_id, agent_traj in traj_dict.items():
        metrics_dict = {}
        timestep = 0.0333333 # For flock. TODO
        tol = 1/timestep
        frames = len(agent_traj)
        print(agent_traj.head(5))
        exit()
        source_x = agent_traj["norm_source"].iloc[0]
        source_z = agent_traj["norm_source"].iloc[1]
        stop_metric = 0 
        dfs = 0 # distance from source
        circling_metric = 0
        afs = 0 # angle from source
        attract_metric = 0
        path_x = []
        path_z = []
        uni_metric = 0
        avoid_metric = 0
        steps = 0
        for i in range(frames):
            pos_x = agent_traj["pos_x"].iloc[i]
            pos_z = agent_traj["pos_z"].iloc[i]
            speed = agent_traj["speed"].iloc[i]
            if speed <= 0.001:
                stop_metric += 1 
            dfs_new = math.dist((pos_x, pos_z),(source_x, source_z)) 
            if abs(dfs - dfs_new) <= 0.0005:
                circling_metric += 1
            dfs = dfs_new
            afs_new = math.degrees(math.atan2(source_z - pos_z, source_x - pos_x)) 
            if abs(afs-afs_new) <= 0.01:
                attract_metric += 1
            afs = afs_new
            if (i % int(tol)) == 0:
                if i == 0:
                    spawn = [pos_x, pos_z]
                else:
                    goal = [pos_x, pos_z]
                    m = (goal[1]-spawn[1])/(goal[0]- spawn[0])
                    b = goal[1] - m * goal[0]
                    optimal_z = m * np.array(path_x) + b
                    deviation = np.linalg.norm(optimal_z - np.array(path_z))
                    uni_metric += deviation
                    steps += 1
                    coefficients = np.polyfit(np.array(path_x), np.array(path_z), 2)
                    polynomial = np.poly1d(coefficients)
                    y_curve = polynomial(np.array(path_x))
                    deviation_curve = np.linalg.norm(y_curve - np.array(path_z))
                    avoid_metric += deviation_curve
                    spawn = goal
                    path_x = []
                    path_z = []
            else:
                path_x.append(pos_x)
                path_z.append(pos_z)
            # exit()
        stop_metric /= frames
        circling_metric /= frames
        attract_metric /= frames
        uni_metric /= steps
        avoid_metric /= steps
        metrics_dict["stop_metric"] = stop_metric
        metrics_dict["circling_metric"] = circling_metric
        metrics_dict["attract_metric"] = attract_metric
        metrics_dict["uni_metric"] = uni_metric
        metrics_dict["avoid_metric"] = avoid_metric
        final_dict[agent_id] = metrics_dict
    exit()

    return final_dict

def get_gt_instructed(c_dict, gt_dict):
    folder_path = f'C:\\PROJECTS\\SocialLandmarks\\SocialLandmarks_Python\\Data\\PythonFiles\\Instructed\\'
    all_gt = os.listdir(folder_path)
    npz_files = [file for file in all_gt if file.lower().endswith('.npz')]
    final_dict = {}
    eval_pred = []
    eval_gt = []
    metric = 0
    for npz_file in npz_files:
        key = npz_file.split(".npz")[0]
        value = int(key.split("class_")[1].split("_")[0])
        pred = gt_dict[c_dict[npz_file]]
        if value == pred:
            res = "match"
            metric += 1
        else:
            res = "fail"
        final_dict[key] = {"pred" : pred, "gt" : value, "success": res}
        eval_pred.append(pred)
        eval_gt.append(value)
    return final_dict, eval_pred, eval_gt, metric

if __name__ ==  '__main__':
    """
    Evaluation with existing dataset is limited to measuring % success in behavior replication.
    The following code then does the following:
        i. 
        ii. Trajectory comparison with inputs/gts given same initial positions.
        WE assume appropriate source placement.
    """

    model_name = "model_final.pth"
    model_type = "pytorch"
    # dataset_name = "Flock"
    # dataset_name = "Zara"
    # dataset_name = "Students"
    dataset_name = "Instructed"
    folder_path = f'C:\\PROJECTS\\SocialLandmarks\\SocialLandmarks_Python\\Data\\PythonFiles\\{dataset_name}\\'  
    x_test, pred_dict = load_inference_data(folder_path, return_dict=True)
    c_batch_size = x_test.shape[0]
    batch_size = 32
    predictions, predicted_labels = model_inference(model_name, model_type, x_test, batch_size)
    combinations, c_dict = decode_labels(predicted_labels, pred_dict)
    # print(c_dict)
    json_path = "C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Experiments\Evaluation"
    json_name = json_path + "\\c_dict.json"
    # with open(json_name, 'w') as json_file:
    #     json.dump(c_dict, json_file, indent=4)
    
    # exit()

    gt_dict = {"1_1": 0, "1_2": 1, "1_3": 2, "1_4": 3, "1_5": 4,
        "2_1": 5, "2_2": 6, "2_3": 7, "2_4": 8, "2_5": 9,
        "3_1": 10, "3_2": 11,"3_3": 12, "3_4": 13, "3_5": 14,
        "4_1": 15, "4_2": 16, "4_3": 17, "4_4": 18, "4_5": 19,
        "5_1": 20, "5_2": 21, "5_3": 22, "5_4": 23, "5_5": 24,
        "0_0": 12, "0_1": 10, "0_2": 11, "0_3": 12, "0_4": 13, "0_5": 14, "0_6": 13,
        "1_0": 2, "2_0": 7, "3_0": 12, "4_0": 17, "5_0": 22, "6_0":17,
        "1_6":3, "2_6":8, "3_6":13, "4_6":18, "5_6":23, "6_6":18,
        "6_1":15, "6_2":16, "6_3":17, "6_4":18, "6_5":19}

    search_dict, json_path = evaluate_trajectories()
    # search_dict, json_path = evaluate_images(query_path = "C:\PROJECTS\SocialLandmarks\Data\Tracking\YOLO\Tracker\Images\Instructed")
    # (search_dict)
    exit()

    final_dict = evaluate_dedicated_metric(traj_directory="C:\PROJECTS\SocialLandmarks\Data\Tracking\YOLO\Tracker\Trajectories\\")
    # final_dict, eval_pred, eval_gt, metric = get_gt_instructed(c_dict, gt_dict)
    json_path = "C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Experiments\Evaluation"
    json_name = json_path + "\\final_dict.json"
    with open(json_name, 'w') as json_file:
        json.dump(final_dict, json_file, indent=4)
    exit()

    # final_dict = {}
    # metric = 0
    # eval_pred = []
    # eval_gt = []
    # for key in c_dict.keys():
    #     key_new = key.split('.npz')[0]
    #     pred = c_dict[key]
    #     pseudo = search_dict[key_new]
    #     pseudo_gt = pseudo.split('IF_')[1].split('_T')[0]
    #     final_dict[key_new] = [pred, pseudo_gt]
    #     pseudo_gt_class = gt_dict[pseudo_gt]
    #     pred_class = gt_dict[pred]
    #     # Measure accuracy:
    #     if pred_class == pseudo_gt_class:
    #         metric += 1
    #     eval_pred.append(pred_class)
    #     eval_gt.append(pseudo_gt_class)

    print(final_dict)
    print("Accuracy: ", (metric/len(c_dict.keys())) * 100, " %")
    # Calculate confusion:
    confusion = confusion_matrix(eval_gt, eval_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=False, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix')
    plt.savefig(f'{json_path}\\confusion_matrix.png')
