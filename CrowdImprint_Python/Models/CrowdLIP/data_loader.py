from imports import *
from helper_functions import *

# python3 C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\CrowdLIP\data_loader.py

def load_data_keras(val, test):
    # LOAD DATA
    if val == True:
        folder_path = 'PythonFiles\\SingleSwitch\\ValidationData'  
    elif test == True:
        folder_path = 'PythonFiles\\SingleSwitchTesting'
    else:
        folder_path = 'PythonFiles\\SingleSwitch\\'  

    file_list = os.listdir(folder_path)
    npz_files = [file for file in file_list if file.endswith('.npz')]

    comb_dict = {"1_1": 0, "1_2": 1, "1_3": 2, "1_4": 3, "1_5": 4,
            "2_1": 5, "2_2": 6, "2_3": 7, "2_4": 8, "2_5": 9,
            "3_1": 10, "3_2": 11,"3_3": 12, "3_4": 13, "3_5": 14,
            "4_1": 15, "4_2": 16, "4_3": 17, "4_4": 18, "4_5": 19,
            "5_1": 20, "5_2": 21, "5_3": 22, "5_4": 23, "5_5": 24,
            "0_0": 12, "0_1": 10, "0_2": 11, "0_3": 12, "0_4": 13, "0_5": 14, "0_6": 13,
            "1_0": 2, "2_0": 7, "3_0": 12, "4_0": 17, "5_0": 22, "6_0":17,
            "1_6":3, "2_6":8, "3_6":13, "4_6":18, "5_6":23, "6_6":18,
            "6_1":15, "6_2":16, "6_3":17, "6_4":18, "6_5":19}

    label_dict = {}
    images_dict = {}

    for npz_file in tqdm(npz_files): #TODO
        class_index = npz_file.split("IF_")[1].split("_T")[0]
        class_type = comb_dict[class_index]
        if class_type in label_dict:
            label_dict[class_type].append(npz_file)
        else:
            label_dict[class_type] = [npz_file]

        # Read image:
        file_path = os.path.join(folder_path, npz_file)
        loaded_data = np.load(file_path)
        array_keys = loaded_data.files
        array_key = array_keys[0]
        array = loaded_data[array_key]
        if (array.dtype != 'float32'):
            print(file_path)
            exit()
        images_dict[npz_file] = array
        
    return label_dict, images_dict

def create_pairs(label_dict, images_dict, n_reps):

    # label_options = list(range(36))

    images_a = []
    images_b = []
    gt = []

    for i in tqdm(range(n_reps)):
        for label, name_list in label_dict.items():
            # Randomly choose match or mismatch:
            match_flag = random.choice([0, 1])
            if match_flag == 0:
                # We have a match:
                gt.append(1)
                # Choose 2 filenames randomly:
                sampled_images = random.sample(name_list, 2)
                image_a = images_dict[sampled_images[0]]
                image_b = images_dict[sampled_images[1]]
                images_a.append(image_a)
                images_b.append(image_b)
                # visualize_image(image_a)
                # visualize_image(image_b)
            else:
                label_options = list(range(25)) #TODO 25
                # Mismatch:
                gt.append(0)
                label_options.remove(label)
                # Randomly choose an image from label:
                base_image = random.choice(name_list)
                image_a = images_dict[base_image]
                images_a.append(image_a)
                # Randomly choose another label:
                label_choice = random.choice(label_options)
                # Random image from the label of choice:
                sampled_image = random.choice(label_dict[label_choice])
                image_b = images_dict[sampled_image]
                images_b.append(image_b)
                # visualize_image(image_a)
                # visualize_image(image_b)
    
    # NORMALIZE DATA
    gt = np.array(gt)
    images_a = scale_to_standard_normal(images_a)
    images_b = scale_to_standard_normal(images_b)

    return images_a, images_b, gt

def load_data(n_reps, val = False, test = False):
    label_dict, images_dict = load_data_keras(val, test)
    # print(len(label_dict.keys()))
    images_a, images_b, gt = create_pairs(label_dict, images_dict, n_reps)

    return images_a, images_b, gt


if __name__ ==  '__main__':
    """
    For training, we sample two images and get their label:
        1: If they represent the same behavior i.e., combination of InFs.
        0: otherwise.
    """
    
    images_a, images_b, gt = load_data(1)
    # print(images_a.shape, images_b.shape)
    # print(gt.shape)
    # print(gt)

    # index = 30
    # print(images_a[index].shape)
    # visualize_image(images_a[index])
    # visualize_image(images_b[index])
    # print(gt[index])
    # exit()