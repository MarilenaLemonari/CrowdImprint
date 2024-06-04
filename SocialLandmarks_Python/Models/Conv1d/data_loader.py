from imports import *
from helper_functions import *

# remove source csv
# python3 C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\Conv1d\data_loader.py

def read_csv_files(csv_directory):
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

def load_traj_data():
    # LOAD TRAJECTORIES
    current_file_dir = "C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\Trajectories"
    name = "\Conv1d" 
    
    csv_directory  = current_file_dir + name + "\\"

    csv_data = read_csv_files(csv_directory)
    n_csvs = len(csv_data)
    dict_list = list(csv_data.items())

    gt_dict = {"1_1": 0, "1_2": 1, "1_3": 2, "1_4": 3, "1_5": 4, "1_6": 5,
        "2_1": 6, "2_2": 7, "2_3": 8, "2_4": 9, "2_5": 10, "2_6": 11,
        "3_1": 12, "3_2": 13,"3_3": 14, "3_4": 15, "3_5": 16, "3_6": 17,
        "4_1": 18, "4_2": 19, "4_3": 20, "4_4": 21, "4_5": 22, "4_6":23,
        "5_1": 24, "5_2": 25, "5_3": 26, "5_4": 27, "5_5": 28, "5_6": 29,
        "6_1": 30, "6_2": 31, "6_3": 32, "6_4": 33, "6_5": 34, "6_6": 35,
        "0_0": 14, "0_1": 12, "0_2": 13, "0_3": 14, "0_4": 15, "0_5": 16, "0_6": 17,
        "1_0": 2, "2_0": 8, "3_0": 14, "4_0": 20, "5_0": 26, "6_0": 32}

    # key, value = dict_list[10]
    # print(key, value)
    # exit()

    trajectories = []
    seq_len = []
    gt = []

    for i in tqdm(range(n_csvs)):
        key, value = dict_list[i]
        # X
        traj_x = value["pos_x"]
        traj_z = value["pos_z"]
        sample_traj = np.column_stack((traj_x, traj_z))
        source_value = value["norm_source"]
        source_feature = np.ones((traj_x.shape[0],)) * source_value
        sample_traj = np.column_stack((sample_traj, source_feature))
        trajectories.append(sample_traj)
        seq_len.append(traj_x.shape[0])

        # Y
        class_index = key.split("IF_")[1].split("_T")[0]
        class_type = gt_dict[class_index]
        gt.append(class_type)
        
    max_seq_len = max(seq_len) 
    X_padded = pad_sequences(trajectories, maxlen=max_seq_len, padding='post', dtype='float32')
    gt = np.array(gt)

    return X_padded, gt, max_seq_len


if __name__ ==  '__main__':
    """
    expl.
    """

    X, labels, max_seq_len  = load_traj_data() 
    print("--------------------------Maximum Sequence Length: ", max_seq_len, " ----------------------------------")   
    print(X.shape, labels.shape)