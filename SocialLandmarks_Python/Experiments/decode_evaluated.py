from imports import *

if __name__ ==  '__main__':
    decoded_dict_folder = "C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Experiments\Evaluation\Other_metrics"
    name = "\search_dict_img.json"
    decoded_dict_path = decoded_dict_folder + name
    with open(decoded_dict_path, 'r') as file:
        image_dict = json.load(file)
    
    gt_dict = {"1_1": 0, "1_2": 1, "1_3": 2, "1_4": 3, "1_5": 4,
    "2_1": 5, "2_2": 6, "2_3": 7, "2_4": 8, "2_5": 9,
    "3_1": 10, "3_2": 11,"3_3": 12, "3_4": 13, "3_5": 14,
    "4_1": 15, "4_2": 16, "4_3": 17, "4_4": 18, "4_5": 19,
    "5_1": 20, "5_2": 21, "5_3": 22, "5_4": 23, "5_5": 24,
    "0_0": 12, "0_1": 10, "0_2": 11, "0_3": 12, "0_4": 13, "0_5": 14, "0_6": 13,
    "1_0": 2, "2_0": 7, "3_0": 12, "4_0": 17, "5_0": 22, "6_0":17,
    "1_6":3, "2_6":8, "3_6":13, "4_6":18, "5_6":23, "6_6":18,
    "6_1":15, "6_2":16, "6_3":17, "6_4":18, "6_5":19}

    total = 0
    accuracy = 0
    for key, value in image_dict.items():
        total += 1
        value_proc = value.split("IF_")[1].split("_T")[0]
        value_new = gt_dict[value_proc]
        key_new = int(key.split("class_")[1].split("_subject")[0])
        if key_new == value_new:
            accuracy += 1
    result = "Accuracy: " + str(accuracy/total * 100) + " %"
    
    image_dict.update({"result" : result})
    with open(decoded_dict_path, 'w') as file:
        json.dump(image_dict, file, indent=4)