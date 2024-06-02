from imports import *
from helper_functions import *

def load_data_keras():
    # LOAD DATA
    folder_path = 'PythonFiles\\SingleSwitch\\'  
    file_list = os.listdir(folder_path)
    npz_files = [file for file in file_list if file.endswith('.npz')]
    loaded_images = []
    field_IDs = []
    gt = []
    gt_dict = {"1_1": 0, "1_2": 1, "1_3": 2, "1_4": 3, "1_5": 4, "1_6": 5,
            "2_1": 6, "2_2": 7, "2_3": 8, "2_4": 9, "2_5": 10, "2_6": 11,
            "3_1": 12, "3_2": 13,"3_3": 14, "3_4": 15, "3_5": 16, "3_6": 17,
            "4_1": 18, "4_2": 19, "4_3": 20, "4_4": 21, "4_5": 22, "4_6":23,
            "5_1": 24, "5_2": 25, "5_3": 26, "5_4": 27, "5_5": 28, "5_6": 29,
            "6_1": 30, "6_2": 31, "6_3": 32, "6_4": 33, "6_5": 34, "6_6": 35,
            "0_0": 14, "0_1": 12, "0_2": 13, "0_3": 14, "0_4": 15, "0_5": 16, "0_6": 17,
            "1_0": 2, "2_0": 8, "3_0": 14, "4_0": 20, "5_0": 26, "6_0": 32}
    class_0 = 0
    class_1 = 0
    class_2 = 0
    class_3 = 0
    class_4 = 0
    class_5 = 0
    class_6 = 0
    class_7 = 0
    class_8 = 0
    class_9 = 0
    class_10 = 0
    class_11 = 0
    class_12 = 0
    class_13 = 0
    class_14 = 0
    class_15 = 0
    class_16 = 0
    class_17 = 0
    class_18 = 0
    class_19 = 0
    class_20 = 0
    class_21 = 0
    class_22 = 0
    class_23 = 0
    class_24 = 0
    class_25 = 0
    class_26 = 0
    class_27 = 0
    class_28 = 0
    class_29 = 0
    class_30 = 0
    class_31 = 0
    class_32 = 0
    class_33 = 0
    class_34 = 0
    class_35 = 0
    # TODO: 2 predictions of 6.
    for npz_file in tqdm(npz_files):
        # name_parts = npz_file.split('_')
        # type_index = name_parts.index("type")
        class_index = npz_file.split("IF_")[1].split("_T")[0]
        field_1 = npz_file.split("IF_")[1].split("_")[0]
        field_2 = npz_file.split("IF_")[1].split("_")[1]
        fields = np.array([field_1, field_2],dtype = np.float32)
        # value_after_type = int(name_parts[type_index + 1])

        # Read image:
        file_path = os.path.join(folder_path, npz_file)
        loaded_data = np.load(file_path)
        array_keys = loaded_data.files
        array_key = array_keys[0]
        array = loaded_data[array_key]
        if (array.dtype != 'float32'):
            print(file_path)
            exit()
        loaded_images.append(array)
        field_IDs.append(fields)
        class_type = gt_dict[class_index]
        gt.append(class_type)

        if class_type == 0:
            class_0 += 1
        elif class_type == 1:
            class_1 += 1
        elif class_type == 2:
            class_2 += 1
        elif class_type == 3:
            class_3 += 1
        elif class_type == 4:
            class_4 += 1
        elif class_type == 5:
            class_5 += 1
        elif class_type == 6:
            class_6 += 1
        elif class_type == 7:
            class_7 += 1
        elif class_type == 8:
            class_8 += 1
        elif class_type == 9:
            class_9 += 1
        elif class_type == 10:
            class_10 += 1
        elif class_type == 11:
            class_11 += 1
        elif class_type == 12:
            class_12 += 1
        elif class_type == 13:
            class_13 += 1
        elif class_type == 14:
            class_14 += 1
        elif class_type == 15:
            class_15 += 1
        elif class_type == 16:
            class_16 += 1
        elif class_type == 17:
            class_17 += 1
        elif class_type == 18:
            class_18 += 1
        elif class_type == 19:
            class_19 += 1
        elif class_type == 20:
            class_20 += 1
        elif class_type == 21:
            class_21 += 1
        elif class_type == 22:
            class_22 += 1
        elif class_type == 23:
            class_23 += 1
        elif class_type == 24:
            class_24 += 1
        elif class_type == 25:
            class_25 += 1
        elif class_type == 26:
            class_26 += 1
        elif class_type == 27:
            class_27 += 1
        elif class_type == 28:
            class_28 += 1
        elif class_type == 29:
            class_29 += 1
        elif class_type == 30:
            class_30 += 1
        elif class_type == 31:
            class_31 += 1
        elif class_type == 32:
            class_32 += 1
        elif class_type == 33:
            class_33 += 1
        elif class_type == 34:
            class_34 += 1
        elif class_type == 35:
            class_35 += 1
        else:
            print("Wrong Type: ",class_type)
            exit()


    # print(class_0,class_1,class_2,class_3,class_4,class_5,class_6,class_7,class_8,class_9,class_10,class_11,class_12,class_13,class_14,class_15,
    #       class_16, class_17, class_18, class_19, class_20, class_21, class_22, class_23, class_24, class_25, class_26, class_27, class_28, class_29, class_30,
    #       class_31, class_32, class_33, class_34, class_35)
    # # 326 332 312 318 364 308 332 314 350 330 336 316 348 338 340 316 338 360 328 340 358 354 330 348 320 348 342 354 308 338 294 340 328 354 312 326
    # exit()

    # NORMALIZE DATA
    gt = np.array(gt)
    #loaded_images = np.array(loaded_images)
    x = scale_to_standard_normal(loaded_images)
    # print(gt.shape, x.shape)
    # exit()

    return x, gt 

if __name__ ==  '__main__':
    images, gt = load_data()
    # index =513
    # print(loaded_images[index].shape)
    # exit()