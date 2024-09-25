from imports import *

def fill_pixel(tol, pixel_x, pixel_z, intensity, image, resolution):
    left = int(max(pixel_x-tol,0))
    right = int(min(pixel_x+tol,resolution))
    top = int(min(pixel_z+tol,resolution))
    bottom = int(max(pixel_z-tol,0))
    image[left:right,bottom:top] = intensity
    return image

def read_csv_files(csv_directory):
    
    csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]

    data_dict = {}
    all_dfs = []
    column_names = ['timestep','pos_x', 'pos_z'] # TODO: this is subject to the csv structure

    row_threshold = 3
    for filename in tqdm(csv_files):
        # Read the CSV file into a pandas DataFrame and assign column names
        df = pd.read_csv(os.path.join(csv_directory, filename), 
            header=None, names=column_names, 
            skiprows=None,
            usecols=[0, 1, 2]) #,sep=';') #TODO: also subject to csv structure
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
        
        tol = 0.5
        
        bound_max += tol
        bound_min -= tol 

        df["pos_x"] = (df['pos_x'] - bound_min) / (bound_max - bound_min) * (1 - 0) 
        df["pos_z"] = (df['pos_z'] - bound_min) / (bound_max - bound_min) * (1 - 0) 

        # TODO: dependent on environment 
        s = len(df["pos_x"])
        source_norm = np.zeros((s))
        source_norm[0] = (0 - bound_min) / (bound_max - bound_min) * (1 - 0)
        df["norm_source"] = list(source_norm)

        data_dict[filename] = df

    return data_dict

def create_centrered_images(key, value, dataset_name, resolution= 32):
    # # default_int = 0.5
    # plt.clf()
    # plt.plot(value["pos_x"], value["pos_z"], c = 'slategrey')
    # plt.scatter(value["pos_x"][0], value["pos_z"][0], c = 'slategrey')
    # plt.scatter(value["norm_source"][0], value["norm_source"][0], c = 'firebrick', marker = '*', s = 200)
    # plt.legend(['Path', 'Spawn', 'Source'])
    # plt.xlabel("Position X")
    # plt.ylabel("Position Z")
    # plt.title("Normalized Path Image")
    # plt.show()
    # # plt.savefig(dataset_name + "\\" + key)
    # exit()
    max_diff_x = max(value["norm_source"][0], 1 - value["norm_source"][0])
    max_diff_z = max(value["norm_source"][1], 1 - value["norm_source"][1])
    # transform from [0,1] to range where source pos is 0:
    source_pos_x = value["norm_source"][0] - value["norm_source"][0]
    source_pos_z = value["norm_source"][1] - value["norm_source"][1]
    pixel_pos_x = value["pos_x"] - value["norm_source"][0]
    pixel_pos_z = value["pos_z"] - value["norm_source"][1]
    # Zoom
    source_pos_x *= ((resolution - 1) / ( 2* max_diff_x))
    source_pos_z *= ((resolution - 1) / ( 2* max_diff_z))
    pixel_pos_x *= ((resolution - 1) / (2 * max_diff_x))
    pixel_pos_z *= ((resolution - 1) / (2 * max_diff_z))
    source_pos_x += ((resolution - 1)/2)
    source_pos_z += ((resolution - 1)/2)
    pixel_pos_x += ((resolution - 1)/2)
    pixel_pos_z += ((resolution - 1)/2)
    image = np.zeros((resolution,resolution), np.float32)
    image_dict = {}
    same_speed_count = 0
    for i in range(len(pixel_pos_x)):
        pixel_x = int(pixel_pos_x[i])
        pixel_z = int(pixel_pos_z[i])
        if i == 0:
            pixel_x_init = pixel_x
            pixel_z_init = pixel_z
            image[pixel_x,pixel_z] = 1
        elif (value["speed"][i] <= 0.001): #== value["speed"][i-1]):
            same_speed_count += 1

        cur_speed = (1- value["speed"][i])*0.6
        if same_speed_count >= 10:
            # tol = 1
            # left = int(max(pixel_x-tol,0))
            # right = int(min(pixel_x+tol,resolution))
            # top = int(min(pixel_z+tol,resolution))
            # bottom = int(max(pixel_z-tol,0))
            # image[left:right,bottom:top] = cur_speed
            image = fill_pixel(3, pixel_x, pixel_z, cur_speed, image, resolution)
            same_speed_count = 0
        else:
            # image[pixel_x,pixel_z] = cur_speed
            if f"{pixel_x}_{pixel_z}" in image_dict.keys():
                image[pixel_x,pixel_z] = 0.85 
            else:
                image[pixel_x,pixel_z] = 0.6
                image_dict[f"{pixel_x}_{pixel_z}"] = 1


    image[pixel_x_init,pixel_z_init] = 1
    image = fill_pixel(2, pixel_x_init, pixel_z_init, 1, image, resolution)
    # tifffile.imwrite(dataset_name + "\\" + key + '_s' + '.tif', image)

    # Place source 
    image[int(source_pos_x), int(source_pos_z)] = 1
    image = fill_pixel(1, int(source_pos_x), int(source_pos_z), 1, image, resolution)
    tifffile.imwrite(dataset_name + "\\" + key + '.tif', image)

def generate_python_files(img_path, npz_path):

    all_files = os.listdir(img_path)
    tif_files = [file for file in all_files if file.lower().endswith('.tif')]

    for tif_file in tqdm(tif_files):
        old_name = tif_file.split('.')[0]
        try:
            image_path = os.path.join(img_path, tif_file)
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            np.savez(f'{npz_path}\{old_name}.npz', image)
        except Exception as e:
            print(f"Error loading image '{tif_file}': {e}")


if __name__ ==  '__main__':

    # Instructions:
    # cd C:\PROJECTS\SocialLandmarks
    # .\.venv\Scripts\activate
    # cd C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Tool
    # python3 data_loader.py

    traj_dir = "C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\Trajectories\CaseStudy"
    img_dir = "C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\Images\CaseStudy"
    npz_dir = "C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\PythonFiles\CaseStudy"

    """
    In general, the timestep limit is 15s. TODO: confirm this for every csv structure.
    1. For name = "\\P2C_20s_Interval\\Arxiepiskopi_0-20" the csv structure is as follows:
        - timestep; x; z; orientation; goal; group; interaction; connectivity (weights)
    2. For name = "\\SL\\Scenario1_friends":
        - timestep, x, z, or_x, or_z
    """
    # name = "\\P2C_20s_Interval\\Arxiepiskopi_0-20"
    name = "\\SL\\Scenario1_friends"
    csv_directory = traj_dir + name + "\\"
    os.makedirs(img_dir + name, exist_ok=True)
    os.makedirs(npz_dir + name, exist_ok=True)

    csv_data = read_csv_files(csv_directory)
    n_csvs = len(csv_data)
    dict_list = list(csv_data.items())

    for i in tqdm(range(n_csvs)):
        key, value = dict_list[i]
        prefix = key.split(".")[0]
        folder_path = img_dir + name

        files = os.listdir(folder_path)
        file_exists = any(file.startswith(prefix) for file in files)
        if file_exists == False:
            empty_predictions = create_centrered_images(prefix, value, folder_path, resolution=64) 
            generate_python_files(folder_path, npz_dir + name)    