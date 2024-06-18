from imports import *
from preprocess_existing import *

# Instructions:
# cd C:\PROJECTS\SocialLandmarks
# Execute .\.venv\Scripts\activate
# cd C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Experiments
# python3 visualize_gen_traj.py

def read_csvs_generated(csv_directory):
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

    return data_dict

def plot_trajectories(key, value, color_dict):
    # gt_dict = {"1_1": 0, "1_2": 1, "1_3": 2, "1_4": 3, "1_5": 4,
    #     "2_1": 5, "2_2": 6, "2_3": 7, "2_4": 8, "2_5": 9,
    #     "3_1": 10, "3_2": 11,"3_3": 12, "3_4": 13, "3_5": 14,
    #     "4_1": 15, "4_2": 16, "4_3": 17, "4_4": 18, "4_5": 19,
    #     "5_1": 20, "5_2": 21, "5_3": 22, "5_4": 23, "5_5": 24,
    #     "0_0": 12, "0_1": 10, "0_2": 11, "0_3": 12, "0_4": 13, "0_5": 14, "0_6": 13,
    #     "1_0": 2, "2_0": 7, "3_0": 12, "4_0": 17, "5_0": 22, "6_0":17,
    #     "1_6":3, "2_6":8, "3_6":13, "4_6":18, "5_6":23, "6_6":18,
    #     "6_1":15, "6_2":16, "6_3":17, "6_4":18, "6_5":19}
    plt.plot(value["pos_x"].to_numpy(), value["pos_z"].to_numpy(), c = color_dict[key])
    plt.plot(value["pos_x"][0], value["pos_z"][0], 'o', c = color_dict[key],  label='_nolegend_')

def read_csv_new(current_file_dir, name):
    csv_directory  = current_file_dir + name + "\\"

    csv_data = read_csvs_generated(csv_directory)
    n_csvs = len(csv_data)
    dict_list = list(csv_data.items())

    # Read combinations for each agent:
    file_name = [f for f in os.listdir(csv_directory) if f.endswith('.json')]
    file_location = csv_directory + f"{file_name[0]}"
    with open(file_location, 'r') as file:
            combo_dict = json.load(file)

    # key, value = dict_list[20]
    # print(key, value)
    # exit()

    color_dict ={}
    color_list = ["slategrey", "firebrick"]
    index = 0
    for i in tqdm(range(n_csvs)):
        key, value = dict_list[i]
        prefix = key.split(".")[0]
        prefix_updated = combo_dict[prefix][2:-2]
        index = len(color_dict)
        color_dict[prefix_updated] = color_list[index]
        empty_predictions = plot_trajectories(prefix_updated, value, color_dict)

    plt.title(f"{name[1:]} Dataset Generated Paths")
    plt.plot(0,0,'k',marker='o', markersize=8)
    legends = list(color_dict.keys())
    plt.legend(legends)
    plt.savefig(csv_directory + "traj_image.png")
    plt.show()

if __name__ ==  '__main__':

    current_file_dir = "C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\Trajectories\Inference"
    name = "\Flock"

    read_csv_new(current_file_dir, name)