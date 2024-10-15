from imports import *

def get_class(combination):
    gt_dict = {"1_1": 0, "1_2": 1, "1_3": 2, "1_4": 3, "1_5": 4,
    "2_1": 5, "2_2": 6, "2_3": 7, "2_4": 8, "2_5": 9,
    "3_1": 10, "3_2": 11,"3_3": 12, "3_4": 13, "3_5": 14,
    "4_1": 15, "4_2": 16, "4_3": 17, "4_4": 18, "4_5": 19,
    "5_1": 20, "5_2": 21, "5_3": 22, "5_4": 23, "5_5": 24,
    "0_0": 12, "0_1": 10, "0_2": 11, "0_3": 12, "0_4": 13, "0_5": 14, "0_6": 13,
    "1_0": 2, "2_0": 7, "3_0": 12, "4_0": 17, "5_0": 22, "6_0":17,
    "1_6":3, "2_6":8, "3_6":13, "4_6":18, "5_6":23, "6_6":18,
    "6_1":15, "6_2":16, "6_3":17, "6_4":18, "6_5":19}
    return gt_dict[combination] 

def make_hist(hist_values, decode_dict, dataset_name):
    hist, bins = np.histogram(hist_values, bins=np.arange(max(hist_values) + 2))
    count_list = np.array(hist.tolist())
    distribution = (count_list * 100) / sum(count_list)
    beh_distr = {"dataset": f"{dataset_name}"}
    tick_names = []
    hist_adj = []
    for index, percentage in enumerate(distribution):
        if percentage == 0:
            continue
        else:
            hist_adj.append(percentage/100)
            tick_names.append(decode_dict[index])
    

    plt.bar(range(len(decode_dict)), hist_adj, tick_label=tick_names, color = 'slategrey', edgecolor = 'firebrick', linewidth=2.5)
    plt.xlabel('Predicted Behaviour Combination')
    plt.ylabel('Predicted Behaviour Frequency (%)')
    plt.title(f'Behaviour Histogram of {dataset_name}')
    plt.savefig(f"Distributions/{dataset_name}_hist.png")
    plt.show()

if __name__ ==  '__main__':

    # Instructions:
    # cd C:\PROJECTS\SocialLandmarks
    # .\.venv\Scripts\activate
    # cd C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Tool
    # python3 build_distributions.py

    # Load predictions:
    pred_loc = "C:\\PROJECTS\\SocialLandmarks\\SocialLandmarks_Python\\Data\\PythonFiles\\CaseStudy\\UMANS\\Scenario3_exhibit"
    json_files = [f for f in os.listdir(pred_loc) if f.endswith('.json')]
    name = "UMANS_sc3"
    counter = 1 
    for json_file in json_files:
        path = os.path.join(pred_loc, json_file)
        hist_values = []
        decode_dict = {}
        with open(path, 'r') as file:
            predictions = json.load(file)
        for idx, combination in predictions.items():
            clss = get_class(combination)
            decode_dict[clss] = combination
            hist_values.append(int(clss))
        print(hist_values)
        make_hist(hist_values, decode_dict, f"{name}_{counter}")
        counter += 1

