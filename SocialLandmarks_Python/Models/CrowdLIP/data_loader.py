from imports import *
from helper_functions import *

# python3 C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\CrowdLIP\data_loader.py

def load_data_keras():
    # LOAD DATA
    folder_path = 'PythonFiles\\SingleSwitch\\'  
    file_list = os.listdir(folder_path)
    npz_files = [file for file in file_list if file.endswith('.npz')]

    comb_dict = {"1_1": 0, "1_2": 1, "1_3": 2, "1_4": 3, "1_5": 4, "1_6": 5,
            "2_1": 6, "2_2": 7, "2_3": 8, "2_4": 9, "2_5": 10, "2_6": 11,
            "3_1": 12, "3_2": 13,"3_3": 14, "3_4": 15, "3_5": 16, "3_6": 17,
            "4_1": 18, "4_2": 19, "4_3": 20, "4_4": 21, "4_5": 22, "4_6":23,
            "5_1": 24, "5_2": 25, "5_3": 26, "5_4": 27, "5_5": 28, "5_6": 29,
            "6_1": 30, "6_2": 31, "6_3": 32, "6_4": 33, "6_5": 34, "6_6": 35,
            "0_0": 14, "0_1": 12, "0_2": 13, "0_3": 14, "0_4": 15, "0_5": 16, "0_6": 17,
            "1_0": 2, "2_0": 8, "3_0": 14, "4_0": 20, "5_0": 26, "6_0": 32}

    label_dict = {}
    images_dict = {}

    for npz_file in tqdm(npz_files):
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
                label_options = list(range(36))
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

def load_data(n_reps):
    label_dict, images_dict = load_data_keras()
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