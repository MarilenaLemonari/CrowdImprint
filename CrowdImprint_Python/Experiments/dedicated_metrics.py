from imports import *
import glob
from itertools import zip_longest
from matplotlib.colors import LinearSegmentedColormap

def get_decoder():
    decoder = {0:"1_1", 1:"1_2",2: "1_3", 3: "1_4", 4: "1_5",
                5: "2_1", 6: "2_2", 7: "2_3", 8: "2_4", 9: "2_5",
                10 :"3_1", 11: "3_2",12: "3_3", 13: "3_4", 14: "3_5",
                15: "4_1", 16: "4_2", 17: "4_3", 18: "4_4", 19: "4_5",
                20: "5_1", 21: "5_2", 22: "5_3", 23: "5_4", 24: "5_5"}
    return decoder

def get_pairs():

    dict_path = ".\\Evaluation\\Revisions"
    files = glob.glob(f"{dict_path}/*.json")
    # prefix_list = list(range(25))
    # prefix = "class_0_"
    # prefix_class = prefix.split("class_")[1].split("_")[0]      

    metrics = np.zeros((len(files), 5, 2))
    pairs_dict = {}
    decoder = get_decoder()
    
    for prefix_class in range(25):
        combo = decoder[int(prefix_class)]
        prefix = f"class_{prefix_class}_"
        spec_metrics = np.zeros((2,5,2))
        spec_metric_instr = np.zeros((5,5))
        spec_metric_gen = np.zeros((10,5))

        for i,file in enumerate(files):
            name = file.split(dict_path)[1].split(".json")[0].split("mod_dict_")[1]

            stop_m = []
            circle_m = []
            attract_m = []
            uni_m = []
            avoid_m = []
            with open(file, "r") as f:
                metrics_dict = json.load(f)

            
            if "instructed" in name.lower():
                j = 0
                for ag, ag_metrics in metrics_dict.items():
                    if prefix in ag:
                        stop_m.append(ag_metrics["stop_metric"])
                        circle_m.append(np.mean(ag_metrics["circle_metric"]))
                        attract_m.append(np.mean(ag_metrics["attract_metric"]))
                        uni_m.append(ag_metrics["uni_raw"])
                        avoid_m.append(ag_metrics["avoid_raw"])

                        spec_metric_instr[j,:] = np.array([ag_metrics["stop_metric"],np.mean(ag_metrics["circle_metric"]),
                                                           np.mean(ag_metrics["attract_metric"]),ag_metrics["uni_raw"],ag_metrics["avoid_raw"]])
                        j += 1
                        
                metrics[i,0,:] = np.array([np.mean(stop_m), np.std(stop_m)])
                metrics[i,1,:] = np.array([np.mean(circle_m), np.std(circle_m)])
                metrics[i,2,:] = np.array([np.mean(attract_m), np.std(attract_m)])
                metrics[i,3,:] = np.array([np.mean(uni_m), np.std(uni_m)])
                metrics[i,4,:] = np.array([np.mean(avoid_m), np.std(avoid_m)])

               
                spec_metrics[0,:,:] = metrics[i,:,:]

            elif combo in name.lower():     
                for j, (ag, ag_metrics) in enumerate(metrics_dict.items()):

                    stop_m.append(ag_metrics["stop_metric"])
                    circle_m.append(np.mean(ag_metrics["circle_metric"]))
                    attract_m.append(np.mean(ag_metrics["attract_metric"]))
                    uni_m.append(ag_metrics["uni_raw"])
                    avoid_m.append(ag_metrics["avoid_raw"])

                    spec_metric_gen[j,:] = np.array([ag_metrics["stop_metric"],np.mean(ag_metrics["circle_metric"]),
                                                           np.mean(ag_metrics["attract_metric"]),ag_metrics["uni_raw"],ag_metrics["avoid_raw"]])
        
                metrics[i,0,:] = np.array([np.mean(stop_m), np.std(stop_m)])
                metrics[i,1,:] = np.array([np.mean(circle_m), np.std(circle_m)])
                metrics[i,2,:] = np.array([np.mean(attract_m), np.std(attract_m)])
                metrics[i,3,:] = np.array([np.mean(uni_m), np.std(uni_m)])
                metrics[i,4,:] = np.array([np.mean(avoid_m), np.std(avoid_m)])

                spec_metrics[1,:,:] = metrics[i,:,:] # 1st row instructed


        pairs_dict[combo] = {"gt": spec_metric_instr, "gen": spec_metric_gen}
    
    return pairs_dict

def get_scenarios():
    dict_path = ".\\Evaluation\\Revisions"
    files = glob.glob(f"{dict_path}/*.json")
    # prefix_list = list(range(25))
    # prefix = "class_0_"
    # prefix_class = prefix.split("class_")[1].split("_")[0]      

    metrics = np.zeros((len(files), 5, 2))
    pairs_dict = {}
    decoder = get_decoder()
    
    for scenario_id in range(1,6):
        scenario_prefix = f"scenario{scenario_id}"
        spec_metrics = np.zeros((2,5,2))
        spec_metric_coll = np.zeros((5,5))
        spec_metric_gen = np.zeros((10,5))

        for i,file in enumerate(files):
            name = file.split(dict_path)[1].split(".json")[0].split("mod_dict_")[1]
            stop_m = []
            circle_m = []
            attract_m = []
            uni_m = []
            avoid_m = []
            with open(file, "r") as f:
                metrics_dict = json.load(f)
            
            if ("scenarioscollected" in name.lower()):
                j = 0
                for ag, ag_metrics in metrics_dict.items():
                    if scenario_prefix in ag:
                        stop_m.append(ag_metrics["stop_metric"])
                        circle_m.append(np.mean(ag_metrics["circle_metric"]))
                        attract_m.append(np.mean(ag_metrics["attract_metric"]))
                        uni_m.append(ag_metrics["uni_raw"])
                        avoid_m.append(ag_metrics["avoid_raw"])

                        spec_metric_coll[j,:] = np.array([ag_metrics["stop_metric"],np.mean(ag_metrics["circle_metric"]),
                                                           np.mean(ag_metrics["attract_metric"]),ag_metrics["uni_raw"],ag_metrics["avoid_raw"]])
                        j += 1
                    
                if len(stop_m) != 0:
                    metrics[i,0,:] = np.array([np.mean(stop_m), np.std(stop_m)])
                    metrics[i,1,:] = np.array([np.mean(circle_m), np.std(circle_m)])
                    metrics[i,2,:] = np.array([np.mean(attract_m), np.std(attract_m)])
                    metrics[i,3,:] = np.array([np.mean(uni_m), np.std(uni_m)])
                    metrics[i,4,:] = np.array([np.mean(avoid_m), np.std(avoid_m)])

                    spec_metrics[0,:,:] = metrics[i,:,:]

            elif scenario_prefix in name.lower():     
                
                for j, (ag, ag_metrics) in enumerate(metrics_dict.items()):
                    stop_m.append(ag_metrics["stop_metric"])
                    circle_m.append(np.mean(ag_metrics["circle_metric"]))
                    attract_m.append(np.mean(ag_metrics["attract_metric"]))
                    uni_m.append(ag_metrics["uni_raw"])
                    avoid_m.append(ag_metrics["avoid_raw"])

                    spec_metric_gen[j,:] = np.array([ag_metrics["stop_metric"],np.mean(ag_metrics["circle_metric"]),
                                                           np.mean(ag_metrics["attract_metric"]),ag_metrics["uni_raw"],ag_metrics["avoid_raw"]])
        
                metrics[i,0,:] = np.array([np.mean(stop_m), np.std(stop_m)])
                metrics[i,1,:] = np.array([np.mean(circle_m), np.std(circle_m)])
                metrics[i,2,:] = np.array([np.mean(attract_m), np.std(attract_m)])
                metrics[i,3,:] = np.array([np.mean(uni_m), np.std(uni_m)])
                metrics[i,4,:] = np.array([np.mean(avoid_m), np.std(avoid_m)])

                spec_metrics[1,:,:] = metrics[i,:,:] 


        pairs_dict[scenario_prefix] = {"gt": spec_metric_coll, "gen": spec_metric_gen}
    
    return pairs_dict
   
def get_pairs_dict():

    dict_path = ".\\Evaluation\\Revisions"
    files = glob.glob(f"{dict_path}/*.json")
    # prefix_list = list(range(25))
    # prefix = "class_0_"
    # prefix_class = prefix.split("class_")[1].split("_")[0]      

    metrics = np.zeros((len(files), 5, 2))
    pairs_dict = {}
    decoder = get_decoder()
    
    for prefix_class in range(25):
        combo = decoder[int(prefix_class)]
        prefix = f"class_{prefix_class}_"
        spec_metrics = np.zeros((2,5,2))
        spec_metric_instr = np.zeros((5,5))
        spec_metric_gen = np.zeros((10,5))

        for i,file in enumerate(files):
            name = file.split(dict_path)[1].split(".json")[0].split("final_dict_")[1]

            stop_m = []
            circle_m = []
            attract_m = []
            uni_m = []
            avoid_m = []
            with open(file, "r") as f:
                metrics_dict = json.load(f)

            
            if "instructed" in name.lower():
                j = 0
                for ag, ag_metrics in metrics_dict.items():
                    if prefix in ag:
                        stop_m.append(ag_metrics["stop_metric"])
                        circle_m.append(ag_metrics["circling_metric"])
                        attract_m.append(ag_metrics["attract_metric"])
                        uni_m.append(ag_metrics["uni_metric"])
                        avoid_m.append(ag_metrics["avoid_metric"])

                        spec_metric_instr[j,:] = np.array([ag_metrics["stop_metric"],ag_metrics["circling_metric"],
                                                           ag_metrics["attract_metric"],ag_metrics["uni_metric"],ag_metrics["avoid_metric"]])
                        j += 1
                        
                metrics[i,0,:] = np.array([np.mean(stop_m), np.std(stop_m)])
                metrics[i,1,:] = np.array([np.mean(circle_m), np.std(circle_m)])
                metrics[i,2,:] = np.array([np.mean(attract_m), np.std(attract_m)])
                metrics[i,3,:] = np.array([np.mean(uni_m), np.std(uni_m)])
                metrics[i,4,:] = np.array([np.mean(avoid_m), np.std(avoid_m)])

               
                spec_metrics[0,:,:] = metrics[i,:,:]

            elif combo in name.lower():     
                for j, (ag, ag_metrics) in enumerate(metrics_dict.items()):
                    stop_m.append(ag_metrics["stop_metric"])
                    circle_m.append(ag_metrics["circling_metric"])
                    attract_m.append(ag_metrics["attract_metric"])
                    uni_m.append(ag_metrics["uni_metric"])
                    avoid_m.append(ag_metrics["avoid_metric"])

                    spec_metric_gen[j,:] = np.array([ag_metrics["stop_metric"],ag_metrics["circling_metric"],
                                                           ag_metrics["attract_metric"],ag_metrics["uni_metric"],ag_metrics["avoid_metric"]])
        
                metrics[i,0,:] = np.array([np.mean(stop_m), np.std(stop_m)])
                metrics[i,1,:] = np.array([np.mean(circle_m), np.std(circle_m)])
                metrics[i,2,:] = np.array([np.mean(attract_m), np.std(attract_m)])
                metrics[i,3,:] = np.array([np.mean(uni_m), np.std(uni_m)])
                metrics[i,4,:] = np.array([np.mean(avoid_m), np.std(avoid_m)])

                spec_metrics[1,:,:] = metrics[i,:,:] # 1st row instructed


        pairs_dict[combo] = {"gt": spec_metric_instr, "gen": spec_metric_gen}
    
    return pairs_dict

def get_scenario_pairs():

    dict_path = ".\\Evaluation\\Revisions"
    files = glob.glob(f"{dict_path}/*.json")
    # prefix_list = list(range(25))
    # prefix = "class_0_"
    # prefix_class = prefix.split("class_")[1].split("_")[0]      

    metrics = np.zeros((len(files), 5, 2))
    pairs_dict = {}
    decoder = get_decoder()
    
    for scenario_id in range(1,6):
        scenario_prefix = f"scenario{scenario_id}"
        spec_metrics = np.zeros((2,5,2))
        spec_metric_coll = np.zeros((5,5))
        spec_metric_gen = np.zeros((10,5))

        for i,file in enumerate(files):
            name = file.split(dict_path)[1].split(".json")[0].split("final_dict_")[1]
            stop_m = []
            circle_m = []
            attract_m = []
            uni_m = []
            avoid_m = []
            with open(file, "r") as f:
                metrics_dict = json.load(f)
            
            if ("scenarioscollected" in name.lower()):
                j = 0
                for ag, ag_metrics in metrics_dict.items():
                    if scenario_prefix in ag:
                        stop_m.append(ag_metrics["stop_metric"])
                        circle_m.append(ag_metrics["circling_metric"])
                        attract_m.append(ag_metrics["attract_metric"])
                        uni_m.append(ag_metrics["uni_metric"])
                        avoid_m.append(ag_metrics["avoid_metric"])

                        spec_metric_coll[j,:] = np.array([ag_metrics["stop_metric"],ag_metrics["circling_metric"],
                                                           ag_metrics["attract_metric"],ag_metrics["uni_metric"],ag_metrics["avoid_metric"]])
                        j += 1
                    
                if len(stop_m) != 0:
                    metrics[i,0,:] = np.array([np.mean(stop_m), np.std(stop_m)])
                    metrics[i,1,:] = np.array([np.mean(circle_m), np.std(circle_m)])
                    metrics[i,2,:] = np.array([np.mean(attract_m), np.std(attract_m)])
                    metrics[i,3,:] = np.array([np.mean(uni_m), np.std(uni_m)])
                    metrics[i,4,:] = np.array([np.mean(avoid_m), np.std(avoid_m)])

                    spec_metrics[0,:,:] = metrics[i,:,:]

            elif scenario_prefix.lower() in name.lower():     
                
                for j, (ag, ag_metrics) in enumerate(metrics_dict.items()):
                    stop_m.append(ag_metrics["stop_metric"])
                    circle_m.append(ag_metrics["circling_metric"])
                    attract_m.append(ag_metrics["attract_metric"])
                    uni_m.append(ag_metrics["uni_metric"])
                    avoid_m.append(ag_metrics["avoid_metric"])

                    spec_metric_gen[j,:] = np.array([ag_metrics["stop_metric"],ag_metrics["circling_metric"],
                                                           ag_metrics["attract_metric"],ag_metrics["uni_metric"],ag_metrics["avoid_metric"]])
        
                metrics[i,0,:] = np.array([np.mean(stop_m), np.std(stop_m)])
                metrics[i,1,:] = np.array([np.mean(circle_m), np.std(circle_m)])
                metrics[i,2,:] = np.array([np.mean(attract_m), np.std(attract_m)])
                metrics[i,3,:] = np.array([np.mean(uni_m), np.std(uni_m)])
                metrics[i,4,:] = np.array([np.mean(avoid_m), np.std(avoid_m)])

                spec_metrics[1,:,:] = metrics[i,:,:] 


        pairs_dict[scenario_prefix] = {"gt": spec_metric_coll, "gen": spec_metric_gen}
    
    return pairs_dict

def get_pairs_scenarios_metrics():
    data_dir = ".\\Evaluation\\Revisions"
    json_files = glob.glob(f"{data_dir}\\*.json")

    gt_combos = "raw_dict_ScenariosCollected.json"
    with open(data_dir+"\\"+gt_combos, "r") as f:
        raw_instructed = json.load(f)

    scenario_list = ["Scenario1_friends", "Scenario2_guard", "Scenario3_exhibit", "Scenario4_atm", "Scenario5_foodcourt"]
    pairs_dict = {}
    for sc_no, scenario in enumerate(scenario_list):
        json_path = data_dir + f"\\raw_dict_{scenario}.json"
        with open(json_path, "r") as f:
            raw_gen = json.load(f)
            metrics_gen = []

        pair_instructed = {}
        instructed_key = f"{scenario}"
        for i_key, i_val in raw_instructed.items():
            if instructed_key.lower() in i_key.lower():
                pair_instructed[i_key] = i_val
    
    
        pairs_dict[scenario] = {"gt": pair_instructed, "gen":raw_gen}
    return pairs_dict

def get_pairs_metrics():
    data_dir = ".\\Evaluation\\Revisions"
    json_files = glob.glob(f"{data_dir}\\*.json")

    gt_combos = "raw_dict_Instructed.json"
    with open(data_dir+"\\"+gt_combos, "r") as f:
        raw_instructed = json.load(f)

    decoder = get_decoder()
    pairs_dict = {}
    for class_no, combo in decoder.items():
        json_path = data_dir + f"\\raw_dict_{combo}.json"
        with open(json_path, "r") as f:
            raw_gen = json.load(f)

        pair_instructed = {}
        instructed_key = f"class_{class_no}_"
        for i_key, i_val in raw_instructed.items():
            if instructed_key in i_key:
                pair_instructed[i_key] = i_val
    
    
        pairs_dict[combo] = {"gt": pair_instructed, "gen":raw_gen}
    return pairs_dict

def create_spec_boxplots(gt, gen, direction, subplot_names):
    fig, ax = plt.subplots(figsize=(8, 6))

    positions_gt = list(range(len(subplot_names)))  # x positions for GT
    positions_gt = [x*2 + 1 for x in positions_gt]
    positions_gen = [p + 0.5 for p in positions_gt]  # small shift for GEN

    # Plot GT and GEN boxplots side by side
    medianprops = dict(color='black', linewidth=2)
    boxprops_gt = dict(facecolor='firebrick', color='black')
    boxprops_gen = dict(facecolor='slategrey', color='black')


    # Draw GT and GEN boxplots
    bp1 = ax.boxplot(gt.T, positions=positions_gt, widths=0.4,
                    patch_artist=True, boxprops=boxprops_gt,
                    medianprops=medianprops)

    bp2 = ax.boxplot(gen.T, positions=positions_gen, widths=0.4,
                    patch_artist=True, boxprops=boxprops_gen,
                    medianprops=medianprops)


    # Axis labels and legend
    ax.set_xticks([p + 0.25 for p in positions_gt])
    ax.set_xticklabels([subplot_names[i] for i in range(len(subplot_names))])
    ax.set_ylabel("Value")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["COLL.", "GEN."], loc="upper right")
    ax.set_title("Collected vs Generated")

    plt.tight_layout()
    plt.savefig(f".\\Evaluation\\Revisions\\Boxplots\\PerMetric\\{direction}.png")
    # plt.show()
    plt.clf()
    plt.close()

def create_boxplot_pairs(gt, gen, title, per_group, subplot_names=["stop_metric", "circle_metric", "attract_metric","uni_metric","avoid_metric"]):
    fig, ax = plt.subplots(figsize=(8, 6))

    positions_gt = [1, 3, 5, 7, 9]   # x positions for GT
    positions_gen = [p + 0.5 for p in positions_gt]  # small shift for GEN

    # Plot GT and GEN boxplots side by side
    medianprops = dict(color='black', linewidth=2)
    boxprops_gt = dict(facecolor='firebrick', color='black')
    boxprops_gen = dict(facecolor='slategrey', color='black')

    # Draw GT and GEN boxplots
    bp1 = ax.boxplot(gt, positions=positions_gt, widths=0.4,
                    patch_artist=True, boxprops=boxprops_gt,
                    medianprops=medianprops)

    bp2 = ax.boxplot(gen, positions=positions_gen, widths=0.4,
                    patch_artist=True, boxprops=boxprops_gen,
                    medianprops=medianprops)


    # Axis labels and legend
    ax.set_xticks([p + 0.25 for p in positions_gt])
    ax.set_xticklabels([subplot_names[i] for i in range(5)])
    ax.set_ylabel("Value")
    ax.set_ylabel("Metrics")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Collected.", "Generated"], loc="lower right")
    ax.set_title("Collected vs Generated")

    plt.tight_layout()
    if per_group==True:
        plt.savefig(f".\\Evaluation\\Revisions\\Figures\\Boxplots\\{title}.png")
    else:
        plt.savefig(f".\\Evaluation\\Revisions\\Figures\\bx_{title}.png")
    plt.close()
    # plt.show()

def create_boxplot_triplets(gt, gen, gen2, title, subplot_names=["stop_metric", "circle_metric", "attract_metric","uni_metric","avoid_metric"]):
    fig, ax = plt.subplots(figsize=(8, 6))

    positions_gt = [1, 3, 5, 7, 9]   # x positions for GT
    positions_gen = [p + 0.4 for p in positions_gt]  # small shift for GEN
    positions_gen2 = [p + 0.4 for p in positions_gen]

    # Plot GT and GEN boxplots side by side
    medianprops = dict(color='black', linewidth=2)
    boxprops_gt = dict(facecolor='steelblue', color='black')
    boxprops_gen = dict(facecolor='slategrey', color='black')
    boxprops_gen2 = dict(facecolor='cadetblue', color='black')

    # Draw GT and GEN boxplots
    bp1 = ax.boxplot(gt, positions=positions_gt, widths=0.4,
                    patch_artist=True, boxprops=boxprops_gt,
                    medianprops=medianprops)

    bp2 = ax.boxplot(gen, positions=positions_gen, widths=0.4,
                    patch_artist=True, boxprops=boxprops_gen,
                    medianprops=medianprops)
    
    bp3 = ax.boxplot(gen2, positions=positions_gen2, widths=0.4,
                    patch_artist=True, boxprops=boxprops_gen2,
                    medianprops=medianprops)


    # Axis labels and legend
    ax.set_xticks([p + 0.4 for p in positions_gt])
    ax.set_xticklabels([subplot_names[i] for i in range(5)])
    ax.set_ylabel("Value")
    ax.set_ylabel("Metrics")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0],  bp3["boxes"][0]], ["Collected.", "Generated", "Generated NS"], loc="lower right")
    ax.set_title("Collected vs Generated")

    plt.tight_layout()
    plt.savefig(f".\\Evaluation\\Revisions\\Figures\\Boxplots\\{title}.png")
    plt.close()
    # plt.show()

def create_boxplot_values(gt, gen, title, per_group, subplot_names=["stop_metric", "circle_metric", "attract_metric","uni_metric","avoid_metric"]):
    fig, ax = plt.subplots(figsize=(8, 6))

    positions_gt = [1, 3, 5, 7, 9]   # x positions for GT
    positions_gen = [p + 0.5 for p in positions_gt]  # small shift for GEN

    # Plot GT and GEN boxplots side by side
    medianprops = dict(color='black', linewidth=2)
    boxprops_gt = dict(facecolor='firebrick', color='black')
    boxprops_gen = dict(facecolor='slategrey', color='black')

    # Draw GT and GEN boxplots
    bp1 = ax.boxplot(gt, positions=positions_gt, widths=0.4,
                    patch_artist=True, boxprops=boxprops_gt,
                    medianprops=medianprops,showfliers=False)

    bp2 = ax.boxplot(gen, positions=positions_gen, widths=0.4,
                    patch_artist=True, boxprops=boxprops_gen,
                    medianprops=medianprops,showfliers=False)


    # Axis labels and legend
    ax.set_xticks([p + 0.25 for p in positions_gt])
    ax.set_xticklabels([subplot_names[i] for i in range(5)])
    ax.set_ylabel("Value")
    ax.set_ylabel("Metrics")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Collected.", "Generated"], loc="upper right")
    ax.set_title("Collected vs Generated")

    plt.tight_layout()
    if per_group==True:
        plt.savefig(f".\\Evaluation\\Revisions\\Figures\\Boxplots\\{title}.png")
    else:
        plt.savefig(f".\\Evaluation\\Revisions\\Figures\\bx_{title}.png")
    plt.close()
    # plt.show()

def create_histogram(gt, gen, title,
                     subplot_names=["stop_metric", "circle_metric", "attract_metric","uni_metric","avoid_metric"],
                     bins=20, density=True):
    """
    Draw 5 subplots; each subplot overlays GT and GEN histograms for the i-th metric.
    Saves to .\\Evaluation\\Revisions\\Boxplots\\{title}.png (same path you used).
    """
    # Ensure iterable-of-iterables
    if len(gt) != len(gen):
        raise ValueError("GT and GEN must have the same number of groups.")
    n = len(subplot_names)

    # Figure: 1 x n subplots, share y for easy visual comparison
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4), sharey=True)

    # If n == 1, axes is a single Axes
    if n == 1:
        axes = [axes]

    for i in range(n):
        gti  = np.asarray(gt[i])
        geni = np.asarray(gen[i])

        # Common bin edges per pair → fair comparison
        all_vals = np.concatenate([gti, geni])
        # handle edge case if all values identical
        if np.all(all_vals == all_vals[0]):
            # make a tiny bin span around the constant value
            epsilon = 1e-9 if all_vals.size == 0 else max(1e-9, 0.01 * (abs(all_vals[0]) + 1))
            bins_i = np.linspace(all_vals[0] - epsilon, all_vals[0] + epsilon, 5)
        else:
            bins_i = np.histogram_bin_edges(all_vals, bins=bins)

        ax = axes[i]
        ax.hist(gti,  bins=bins_i, alpha=0.55, color='firebrick',  edgecolor='black',
                label='Collected' if i == 0 else None, density=density)
        ax.hist(geni, bins=bins_i, alpha=0.55, color='slategrey', edgecolor='black',
                label='Generated'  if i == 0 else None, density=density)

        # # Optional: draw medians as black lines
        # if gti.size:
        #     ax.axvline(np.median(gti),  color='black', linewidth=1, linestyle='-')
        # if geni.size:
        #     ax.axvline(np.median(geni), color='black', linewidth=1, linestyle='-')

        ax.set_title(subplot_names[i])
        if i == 0:
            ax.set_ylabel("Density" if density else "Count")
        ax.set_xlabel("Value")

    # Common title + legend
    fig.suptitle("Collected vs Generated", y=0.98)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir = ".\\Evaluation\\Revisions\\Figures\\"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{title}.png")
    plt.savefig(out_path, dpi=200)
    plt.clf()
    plt.close()
    # print(out_path)  # optional feedback

def get_metrics_dict():
    name_dict = {"stop": 0, "circle":1, "attract":2, "continue":3, "uni":3, "avoid":4}
    return name_dict

def get_metric_class():
    metric_class = {"stop":5,"circle":3, "attract":2, "continue":1, "uni":1, "avoid":4}
    return metric_class

def plot_radar(scores,group_label):
    num_vars = 5
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    # scores = np.array([
    #     [0.8, 0.7, 0.9, 0.6, 0.5],
    #     [0.6, 0.5, 0.7, 0.8, 0.9],
    # ])
    groups = ['STOP', 'CIRCLE', 'ATTRACT', 'CONTINUE', 'AVOID']
    groups_alt = ['Speed', 'DFS', 'AFS', 'DFOP', 'DFC']
    label_list = ["Collected", "Generated", "Generated NS"]
    color_dict = {"Collected":"firebrick", "Generated":"slategrey", "Generated NS":"steelblue"}

    # repeat first angle for closing the polygon
    angles += angles[:1]
    fig, ax = plt.subplots(subplot_kw=dict(polar=True))
    
    for c,s in enumerate(scores):
        values = s.tolist() + s[:1].tolist()
        ax.plot(angles, values, linewidth=2, color=color_dict[label_list[c]], label=label_list[c])
        ax.fill(angles, values, color='lightgrey', alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(groups)
    ax.legend(loc = "lower right")
    ax.set_title('Per-group Collected vs Generated score pattern.')
    plt.savefig(f'.\\Evaluation\\Revisions\\Figures\\RadarPlots\\rp_{group_label}.png')
    #plt.show()
    plt.clf()
    plt.close()

def make_heatmaps_scores(pairs_dict,num_metrics=5):

    num_rows = len(pairs_dict.keys())
    heat_matrix_gt = np.zeros((num_rows, num_metrics))
    heat_matrix_gen = np.zeros((num_rows, num_metrics))

    for row, (combo, combo_dict) in enumerate(pairs_dict.items()):
        coll_values = combo_dict["gt"]
        gen_values = combo_dict["gen"]

        gen_col = np.mean(gen_values, axis = 0)
        coll_col = np.mean(coll_values, axis = 0)

        heat_matrix_gen[row, :] = gen_col
        heat_matrix_gt[row, :] = coll_col
    

    # Normalize:
    heat_matrix_gt = norm_cols(heat_matrix_gt)
    heat_matrix_gen = norm_cols(heat_matrix_gen)

    # Draw heatmap:
    # colors = ["firebrick","slategrey","steelblue"]
    # cmap = LinearSegmentedColormap.from_list("my_cmap", colors)
    cmap = "GnBu"
    plt.imshow(heat_matrix_gt, aspect='auto',cmap=cmap,vmin=0, vmax=1)
    plt.xlabel("Collected")
    plt.colorbar()
    plt.savefig('.\\Evaluation\\Revisions\\Figures\\hm_gt.png')
    #plt.show()
    plt.clf()
    plt.imshow(heat_matrix_gen, aspect='auto',cmap=cmap,vmin=0, vmax=1)
    plt.xlabel("Generated")
    plt.colorbar()
    plt.savefig('.\\Evaluation\\Revisions\\Figures\\hm_gen.png')
    #plt.show()
    plt.clf()

def make_radarplots_scores(pairs_dict):

    for plot_id, (combo, combo_dict) in enumerate(pairs_dict.items()):
        coll_values = combo_dict["gt"]
        gen_values = combo_dict["gen"]

        # Normalize:
        gen_values = norm_cols(gen_values)
        coll_values = norm_cols(coll_values)

        gen_avg = np.mean(gen_values, axis = 0) #average over agents
        coll_avg = np.mean(coll_values, axis = 0)

        # Normalize:
        coll_values = norm_cols(coll_values)
        gen_values = norm_cols(gen_values)

        plot_radar([coll_avg,gen_avg], group_label=combo)

def make_boxplot_scores(pairs_dict, per_group):

    if per_group==True:
        for plot_id, (combo, combo_dict) in enumerate(pairs_dict.items()):
            coll_values = combo_dict["gt"]
            gen_values = combo_dict["gen"]

            # Normalize:
            coll_values = norm_cols(coll_values)
            gen_values = norm_cols(gen_values)

            create_boxplot_pairs(coll_values, gen_values, title=combo, per_group=per_group)
    
    if per_group==False:
        all_coll_values = []
        all_gen_values = []

        for plot_id, (combo, combo_dict) in enumerate(pairs_dict.items()):
            coll_values = combo_dict["gt"]
            gen_values = combo_dict["gen"]

            # all_coll_values.append(coll_values)
            # all_gen_values.append(gen_values)
            all_coll_values.append(np.mean(coll_values, axis=0))
            all_gen_values.append(np.mean(gen_values, axis = 0))

        all_coll = np.array(all_coll_values).reshape(-1, 5) 
        all_gen = np.array(all_gen_values).reshape(-1, 5) 

        # Normalize:
        all_coll = norm_cols(all_coll)
        all_gen = norm_cols(all_gen)

        create_boxplot_pairs(all_coll, all_gen, title="all_scores", per_group=per_group)

def make_histogram_scores(pairs_dict):
    all_coll_values = []
    all_gen_values = []

    for plot_id, (combo, combo_dict) in enumerate(pairs_dict.items()):
        coll_values = combo_dict["gt"]
        gen_values = combo_dict["gen"]

        all_coll_values.append(coll_values)
        all_gen_values.append(gen_values)
        # all_coll_values.append(np.mean(coll_values, axis=0))
        # all_gen_values.append(np.mean(gen_values, axis = 0))


    all_coll = np.array(all_coll_values).reshape(-1, 5) 
    all_gen = np.array(all_gen_values).reshape(-1, 5) 

    # Normalize:
    all_coll = norm_cols(all_coll)
    all_gen = norm_cols(all_gen)

    create_histogram(all_coll.T, all_gen.T,"hs_all_scores")
 
def norm_cols(arr, cols=(1, 2)):
    arr_norm = arr.copy().astype(float)   # avoid modifying original
    
    for c in cols:
        col = arr[:, c]
        min_val = col.min()
        max_val = col.max()
        range_val = max_val - min_val
        
        if range_val == 0:
            range_val = 1  # avoid division by zero
        
        arr_norm[:, c] = (col - min_val) / range_val
    
    return arr_norm

def norm_arr(arr):
    mins = arr.min(axis=0)      # shape (5,)
    maxs = arr.max(axis=0)

    # avoid division by zero if a column is constant
    ranges = maxs - mins
    ranges[ranges == 0] = 1

    arr_norm = (arr - mins) / ranges
    return arr_norm

def make_heatmaps_values(pairs_dict):

    n_classes = len(pairs_dict.keys())
    all_coll_arr = np.zeros((n_classes,5))
    all_gen_arr = np.zeros((n_classes,5))

    for n, (key, value) in enumerate(pairs_dict.items()):
        gen_dict = value["gen"]
        coll_dict = value["gt"]
        gen_arr = np.zeros((len(gen_dict),5))
        for i, ag_i in enumerate(gen_dict):
            gen_arr[i,0] = np.mean(np.array(gen_dict[ag_i]["stop_raw"]))
            gen_arr[i,1] = np.mean(np.array(gen_dict[ag_i]["circle_raw"]))
            gen_arr[i,2] = np.mean(np.array(gen_dict[ag_i]["attract_raw"]))
            gen_arr[i,3] = np.mean(np.array(gen_dict[ag_i]["uni_raw"]))
            gen_arr[i,4] = np.mean(np.array(gen_dict[ag_i]["avoid_raw"]))
        gen_avg = np.mean(gen_arr, axis=0)
        
        coll_arr = np.zeros((len(coll_dict),5))
        for i, ag_i in enumerate(coll_dict):
            coll_arr[i,0] = np.mean(np.array(coll_dict[ag_i]["stop_raw"]))
            coll_arr[i,1] = np.mean(np.array(coll_dict[ag_i]["circle_raw"]))
            coll_arr[i,2] = np.mean(np.array(coll_dict[ag_i]["attract_raw"]))
            coll_arr[i,3] = np.mean(np.array(coll_dict[ag_i]["uni_raw"]))
            coll_arr[i,4] = np.mean(np.array(coll_dict[ag_i]["avoid_raw"]))
        coll_avg = np.mean(coll_arr, axis=0)
        
        # all_coll_arr.append(coll_arr)
        # all_gen_arr.append(gen_arr)
        all_coll_arr[n,:] = coll_avg
        all_gen_arr[n,:] = gen_avg
    
    # Normalize:
    norm_gen = norm_arr(all_gen_arr)
    norm_coll = norm_arr(all_coll_arr)

    # Draw heatmap:
    # colors = ["firebrick","slategrey","steelblue"]
    # cmap = LinearSegmentedColormap.from_list("my_cmap", colors)
    cmap = "GnBu"
    plt.imshow(norm_coll, aspect='auto',cmap=cmap)
    plt.xlabel("Collected")
    plt.colorbar()
    plt.savefig('.\\Evaluation\\Revisions\\Figures\\hm_gt_values.png')
    #plt.show()
    plt.clf()
    plt.imshow(norm_gen, aspect='auto',cmap=cmap)
    plt.xlabel("Generated")
    plt.colorbar()
    plt.savefig('.\\Evaluation\\Revisions\\Figures\\hm_gen_values.png')
    #plt.show()
    plt.clf()
    
def make_boxplot_values(pairs_dict, per_group):

    if per_group == False:
        all_coll_arr = []
        all_gen_arr = []

        for n, (key, value) in enumerate(pairs_dict.items()):
            gen_dict = value["gen"]
            coll_dict = value["gt"]

            gen_arr = np.zeros((len(gen_dict),5))
            for i, ag_i in enumerate(gen_dict):
                gen_arr[i,0] = np.mean(np.array(gen_dict[ag_i]["stop_raw"]))
                gen_arr[i,1] = np.mean(np.array(gen_dict[ag_i]["circle_raw"]))
                gen_arr[i,2] = np.mean(np.array(gen_dict[ag_i]["attract_raw"]))
                gen_arr[i,3] = np.mean(np.array(gen_dict[ag_i]["uni_raw"]))
                gen_arr[i,4] = np.mean(np.array(gen_dict[ag_i]["avoid_raw"]))
            gen_avg = np.mean(gen_arr, axis=0)
            
            coll_arr = np.zeros((len(coll_dict),5))
            for i, ag_i in enumerate(coll_dict):
                coll_arr[i,0] = np.mean(np.array(coll_dict[ag_i]["stop_raw"]))
                coll_arr[i,1] = np.mean(np.array(coll_dict[ag_i]["circle_raw"]))
                coll_arr[i,2] = np.mean(np.array(coll_dict[ag_i]["attract_raw"]))
                coll_arr[i,3] = np.mean(np.array(coll_dict[ag_i]["uni_raw"]))
                coll_arr[i,4] = np.mean(np.array(coll_dict[ag_i]["avoid_raw"]))
            coll_avg = np.mean(coll_arr, axis=0)
            
            # all_coll_arr.append(coll_arr)
            # all_gen_arr.append(gen_arr)
            all_coll_arr.append(coll_avg)
            all_gen_arr.append(gen_avg)
        
        all_coll = np.array(all_coll_arr).reshape(-1, 5) 
        all_gen = np.array(all_gen_arr).reshape(-1, 5)

        # Normalize:
        all_coll = norm_arr(all_coll)
        all_gen = norm_arr(all_gen)

        create_boxplot_values(all_coll, all_gen,"all_values",per_group)
    
    else:

        for n, (key, value) in enumerate(pairs_dict.items()):
            gen_dict = value["gen"]
            coll_dict = value["gt"]

            gen_arr = np.zeros((len(gen_dict),5))
            for i, ag_i in enumerate(gen_dict):
                gen_arr[i,0] = np.mean(np.array(gen_dict[ag_i]["stop_raw"]))
                gen_arr[i,1] = np.mean(np.array(gen_dict[ag_i]["circle_raw"]))
                gen_arr[i,2] = np.mean(np.array(gen_dict[ag_i]["attract_raw"]))
                gen_arr[i,3] = np.mean(np.array(gen_dict[ag_i]["uni_raw"]))
                gen_arr[i,4] = np.mean(np.array(gen_dict[ag_i]["avoid_raw"]))
            gen_avg = np.mean(gen_arr, axis=0)
            
            coll_arr = np.zeros((len(coll_dict),5))
            for i, ag_i in enumerate(coll_dict):
                coll_arr[i,0] = np.mean(np.array(coll_dict[ag_i]["stop_raw"]))
                coll_arr[i,1] = np.mean(np.array(coll_dict[ag_i]["circle_raw"]))
                coll_arr[i,2] = np.mean(np.array(coll_dict[ag_i]["attract_raw"]))
                coll_arr[i,3] = np.mean(np.array(coll_dict[ag_i]["uni_raw"]))
                coll_arr[i,4] = np.mean(np.array(coll_dict[ag_i]["avoid_raw"]))
            coll_avg = np.mean(coll_arr, axis=0)
            
            # Normalize:
            coll = norm_arr(coll_arr)
            gen = norm_arr(gen_arr)

            # gen_avg = np.mean(gen, axis=0)
            # coll_avg = np.mean(coll, axis=0)

            create_boxplot_values(coll, gen,f"{key}_values",per_group)

def make_radarplots_values(pairs_dict):

    for n, (key, value) in enumerate(pairs_dict.items()):
        gen_dict = value["gen"]
        coll_dict = value["gt"]

        gen_arr = np.zeros((len(gen_dict),5))
        for i, ag_i in enumerate(gen_dict):
            gen_arr[i,0] = np.mean(np.array(gen_dict[ag_i]["stop_raw"]))
            gen_arr[i,1] = np.mean(np.array(gen_dict[ag_i]["circle_raw"]))
            gen_arr[i,2] = np.mean(np.array(gen_dict[ag_i]["attract_raw"]))
            gen_arr[i,3] = np.mean(np.array(gen_dict[ag_i]["uni_raw"]))
            gen_arr[i,4] = np.mean(np.array(gen_dict[ag_i]["avoid_raw"]))
        
        
        coll_arr = np.zeros((len(coll_dict),5))
        for i, ag_i in enumerate(coll_dict):
            coll_arr[i,0] = np.mean(np.array(coll_dict[ag_i]["stop_raw"]))
            coll_arr[i,1] = np.mean(np.array(coll_dict[ag_i]["circle_raw"]))
            coll_arr[i,2] = np.mean(np.array(coll_dict[ag_i]["attract_raw"]))
            coll_arr[i,3] = np.mean(np.array(coll_dict[ag_i]["uni_raw"]))
            coll_arr[i,4] = np.mean(np.array(coll_dict[ag_i]["avoid_raw"]))
        
    
        # Normalize:
        coll = norm_arr(coll_arr)
        gen = norm_arr(gen_arr)

        gen_avg = np.mean(gen, axis=0)
        coll_avg = np.mean(coll, axis=0)

        plot_radar([coll_avg,gen_avg], group_label=f"{key}_values")

def make_histogram_values(pairs_dict):
    
    all_coll_arr = []
    all_gen_arr = []

    for n, (key, value) in enumerate(pairs_dict.items()):
        gen_dict = value["gen"]
        coll_dict = value["gt"]

        gen_arr = np.zeros((len(gen_dict),5))
        for i, ag_i in enumerate(gen_dict):
            gen_arr[i,0] = np.mean(np.array(gen_dict[ag_i]["stop_raw"]))
            gen_arr[i,1] = np.mean(np.array(gen_dict[ag_i]["circle_raw"]))
            gen_arr[i,2] = np.mean(np.array(gen_dict[ag_i]["attract_raw"]))
            gen_arr[i,3] = np.mean(np.array(gen_dict[ag_i]["uni_raw"]))
            gen_arr[i,4] = np.mean(np.array(gen_dict[ag_i]["avoid_raw"]))
        gen_avg = np.mean(gen_arr, axis=0)
        
        coll_arr = np.zeros((len(coll_dict),5))
        for i, ag_i in enumerate(coll_dict):
            coll_arr[i,0] = np.mean(np.array(coll_dict[ag_i]["stop_raw"]))
            coll_arr[i,1] = np.mean(np.array(coll_dict[ag_i]["circle_raw"]))
            coll_arr[i,2] = np.mean(np.array(coll_dict[ag_i]["attract_raw"]))
            coll_arr[i,3] = np.mean(np.array(coll_dict[ag_i]["uni_raw"]))
            coll_arr[i,4] = np.mean(np.array(coll_dict[ag_i]["avoid_raw"]))
        coll_avg = np.mean(coll_arr, axis=0)
        
        all_coll_arr.append(coll_arr)
        all_gen_arr.append(gen_arr)
        # all_coll_arr.append(coll_avg)
        # all_gen_arr.append(gen_avg)
    
    all_coll = np.array(all_coll_arr).reshape(-1, 5) 
    all_gen = np.array(all_gen_arr).reshape(-1, 5)

    # Normalize:
    all_coll = norm_arr(all_coll)
    all_gen = norm_arr(all_gen)
        
    create_histogram(all_coll.T,all_gen.T,"hs_all_metrics")
        
def main():
    is_scores = None
    name = "scenarios" #instructed, scenarios
    # direction = "all"

    if is_scores == True:
        if name == "scenarios":
            pairs_dict = get_scenario_pairs()
        else:
            pairs_dict = get_pairs_dict()  
    elif is_scores == False:
        if name =="scenarios":
            pairs_dict = get_pairs_scenarios_metrics()
        else:
            pairs_dict = get_pairs_metrics()
    else:
        if name == "scenarios":
            pairs_dict = get_scenarios()
        else:
            pairs_dict = get_pairs()

    
    make_heatmaps_scores(pairs_dict) 
    make_boxplot_scores(pairs_dict, per_group=False)
    make_histogram_scores(pairs_dict)

    make_radarplots_scores(pairs_dict)
    make_boxplot_scores(pairs_dict, per_group = True)

    exit()
    if is_scores == True:
        # Heatmaps:
        make_heatmaps_scores(pairs_dict) # of final_dict not raw values
        # TODO: make_heatmaps_scores(pairs_dict) #noswitch
        # TODO: make_heatmaps_scores(pairs_dict) #2switch

        # Boxplot of all classes:
        make_boxplot_scores(pairs_dict, per_group=False)

        # Histogram:
        make_histogram_scores(pairs_dict)

        # Per group --------------------------

        # Spider plots:
        make_radarplots_scores(pairs_dict)

        # Boxplots:
        make_boxplot_scores(pairs_dict, per_group = True)

    else:
        # Raw metric values ========================== is_Scores=False
        
        make_heatmaps_values(pairs_dict)
        make_boxplot_values(pairs_dict, per_group=False)
        make_histogram_values(pairs_dict)

        make_boxplot_values(pairs_dict, per_group=True)
        make_radarplots_values(pairs_dict)


    '''
    exit()
    if direction == "all":
        gt_scores = []
        gen_scores = []
        for key, metrics_value in pairs_dict.items():
            print(key)
            gt = metrics_value["gt"]
            gt_means = np.mean(gt, axis=0)
            gt_scores.append(gt_means)
            
            gen = metrics_value["gen"] # col is metrics
            gen_means = np.mean(gen, axis=0)
            gen_scores.append(gen_means)
            
            # plot_radar([gen_means, gt_means])
            create_boxplot_pairs(gt, gen, title=name+"_"+key)
            
        
        exit()
        
        if name == "scenarios":
            gt_list = [] # 5, N_classes with direction
            gen_list = []
            subplot_names = []
            for key, metrics_value in pairs_dict.items():
                gt = metrics_value["gt"]
                gen = metrics_value["gen"]
                subplot_names.append(key)
                gt_list.append(gt)
                gen_list.append(gen)

                create_histogram(gt.T,gen.T,title=name+"_"+key)

            # gt=np.transpose(np.array(gt_list), (2, 1, 0))
            # gen =np.transpose(np.array(gen_list), (2, 1, 0)) #(metr,num, sc)

    else:
        if name == "scenarios":
            print("ERROR: Cannot do metric-specific boxplot for scenario.")
            exit()
        else:
            gt_list = [] # 5, N_classes with direction
            gen_list = []
            subplot_names = []
            for key, metrics_value in pairs_dict.items():
                gt = metrics_value["gt"]
                gen = metrics_value["gen"]
                name_dict = get_metrics_dict()
                column = name_dict[direction]
                metric_class = get_metric_class()
                if str(metric_class[direction]) in key:
                    gt_list.append(gt[:,column])
                    gen_list.append(gen[:,column])
                    subplot_names.append(key)

            gt=np.array(gt_list)
            gen = np.array(gen_list)
    
            create_spec_boxplots(gt,gen,name+"_"+direction, direction,subplot_names)
    '''

def find_metric_thesholds():
    data_dir = ".\\Evaluation\\Revisions"
    json_files = glob.glob(f"{data_dir}\\*.json")
    print(len(json_files))

    gt_combos = "raw_dict_Instructed.json"
    with open(data_dir+"\\"+gt_combos, "r") as f:
        raw_instructed = json.load(f)

    decoder = get_decoder()
    for class_no, combo in decoder.items():
        json_path = data_dir + f"\\raw_dict_{combo}.json"
        with open(json_path, "r") as f:
            raw_gen = json.load(f)
        
        if combo == "5_5":
            metric_gen = []
            metric_gt = []
            for i in range(10):
                c_list = raw_gen[f"agent_{i+1}.csv"]["stop_raw"]
                metric_gen.append(c_list) #(10, num_frames)
            for j in range(5):
                metric_gt.append(raw_instructed[f"class_{class_no}_subject{j+1}.csv"]["stop_raw"]) #(5, num_frames)
            
            
        if combo == "3_3":
            metric_gen = []
            metric_gt = []
            for i in range(10):
                c_list = raw_gen[f"agent_{i+1}.csv"]["circle_raw"]
                metric_gen.append(c_list) #(10, num_frames)
            for j in range(5):
                metric_gt.append(raw_instructed[f"class_{class_no}_subject{j+1}.csv"]["circle_raw"]) #(5, num_frames)

        if combo == "2_2":
            metric_gen = []
            metric_gt = []
            for i in range(10):
                c_list = raw_gen[f"agent_{i+1}.csv"]["attract_raw"]
                metric_gen.append(c_list) #(10, num_frames)
            for j in range(5):
                metric_gt.append(raw_instructed[f"class_{class_no}_subject{j+1}.csv"]["attract_raw"]) #(5, num_frames)
        
        if combo == "1_1":
            metric_gen = []
            metric_gt = []
            for i in range(10):
                c_list = raw_gen[f"agent_{i+1}.csv"]["uni_raw"]
                metric_gen.append(c_list) #(10, num_frames)
            for j in range(5):
                metric_gt.append(raw_instructed[f"class_{class_no}_subject{j+1}.csv"]["uni_raw"]) #(5, num_frames)

        if combo == "4_4":
            metric_gen = []
            metric_gt = []
            for i in range(10):
                c_list = raw_gen[f"agent_{i+1}.csv"]["avoid_raw"]
                metric_gen.append(c_list) #(10, num_frames)
            for j in range(5):
                metric_gt.append(raw_instructed[f"class_{class_no}_subject{j+1}.csv"]["avoid_raw"]) #(5, num_frames)

            medianprops = dict(color='black', linewidth=2)
            boxprops_gt = dict(facecolor='firebrick', color='black')
            boxprops_gen = dict(facecolor='slategrey', color='black')

            # Draw GT and GEN boxplots
            bp = plt.boxplot(metric_gt, widths=0.4,
                            patch_artist=True, boxprops=boxprops_gt,
                            medianprops=medianprops, showfliers=False)
            plt.show()

            medians = [line.get_ydata()[0] for line in bp['medians']]
            print(np.mean(medians))
            exit()

def perform_comparisons():
    
    data_dir = ".\\Evaluation\\Revisions"
    json_files = glob.glob(f"{data_dir}\\*.json")
    print(len(json_files))

    
    scenario_list = ["Scenario1_friends", "Scenario2_guard", "Scenario3_exhibit", "Scenario4_atm", "Scenario5_foodcourt"]
    error_gen, error_gen2 = [], [] 
    scores_gt_list, scores_gen_list, scores_gen2_list = [], [], [] 
    for s, scenario in enumerate(scenario_list):
        scores = np.zeros((3,5))
        for file in json_files:
            if "scenarioscollected" in file.lower():
                # gt
                with open(file, "r") as f:
                    metrics_dict = json.load(f)
                stop_m, circle_m, attract_m, uni_m, avoid_m = [], [],[],[],[]
                for keys, ag_metrics in metrics_dict.items():
                    if scenario.lower() in keys.lower():
                        stop_m.append(ag_metrics["stop_metric"])
                        circle_m.append(np.mean(ag_metrics["circle_metric"]))
                        attract_m.append(np.mean(ag_metrics["attract_metric"]))
                        uni_m.append(ag_metrics["uni_raw"])
                        avoid_m.append(ag_metrics["avoid_raw"])

                scores_gt = np.array([stop_m, circle_m, attract_m, uni_m, avoid_m]).T
                scores[0,:] = np.array([np.mean(stop_m),np.mean(circle_m),np.mean(attract_m),np.mean(uni_m),np.mean(avoid_m)])

                        
            elif ("_ns" in file.lower()):
                # no switch
                with open(file, "r") as f:
                    metrics_dict = json.load(f)
                stop_m, circle_m, attract_m, uni_m, avoid_m = [], [],[],[],[]

                for agent, ag_metrics in metrics_dict.items():
                    stop_m.append(ag_metrics["stop_metric"])
                    circle_m.append(np.mean(ag_metrics["circle_metric"]))
                    attract_m.append(np.mean(ag_metrics["attract_metric"]))
                    uni_m.append(ag_metrics["uni_raw"])
                    avoid_m.append(ag_metrics["avoid_raw"])

                scores_gen2 = np.array([stop_m, circle_m, attract_m, uni_m, avoid_m]).T
                scores[2,:] = np.array([np.mean(stop_m),np.mean(circle_m),np.mean(attract_m),np.mean(uni_m),np.mean(avoid_m)])

            elif ("_ns" not in file.lower()):
                #ours
                with open(file, "r") as f:
                    metrics_dict = json.load(f)
                stop_m, circle_m, attract_m, uni_m, avoid_m = [], [],[],[],[]
                for agent, ag_metrics in metrics_dict.items():
                    stop_m.append(ag_metrics["stop_metric"])
                    circle_m.append(np.mean(ag_metrics["circle_metric"]))
                    attract_m.append(np.mean(ag_metrics["attract_metric"]))
                    uni_m.append(ag_metrics["uni_raw"])
                    avoid_m.append(ag_metrics["avoid_raw"])

                scores_gen = np.array([stop_m, circle_m, attract_m, uni_m, avoid_m]).T
                scores[1,:] = np.array([np.mean(stop_m),np.mean(circle_m),np.mean(attract_m),np.mean(uni_m),np.mean(avoid_m)])
        
        diff_gen = abs(np.mean(scores_gt, axis=0) - np.mean(scores_gen, axis=0))
        diff_gen2 = abs(np.mean(scores_gt, axis=0) - np.mean(scores_gen2, axis=0))
        error_gen.append(diff_gen)
        error_gen2.append(diff_gen2)
        

        # scores_gt_list.append(scores_gt)
        # scores_gen_list.append(scores_gen)
        # scores_gen2_list.append(scores_gen2)

        # scores_gt = norm_cols(scores_gt)
        # scores_gen = norm_cols(scores_gen)
        # scores_gen2 = norm_cols(scores_gen2)
        #create_boxplot_triplets(scores_gt,scores_gen, scores_gen2,scenario)

        # scores_gt = norm_cols(scores_gt)
        # scores_gen = norm_cols(scores_gen)
        # scores_gen2 = norm_cols(scores_gen2)
        # arr = [np.mean(scores_gt,axis=0), np.mean(scores_gen,axis=0),np.mean(scores_gen2,axis=0)]
        # plot_radar(arr, scenario)

        # create_histogram(scores_gen.T, scores_gen2.T, "hey")
    

    print("Error from ours: ", np.mean(error_gen), np.std(error_gen))
    print("Error from NS: ",np.mean(error_gen2), np.mean(error_gen2))
    create_boxplot_triplets(error_gen2, error_gen, error_gen2, "errors")
    
    # cmap = "GnBu"
    # plt.imshow(error_gen, aspect='auto',cmap=cmap,vmin=0, vmax=0.6)
    # plt.xlabel("Diff w/ours")
    # plt.colorbar()
    # plt.savefig('.\\Evaluation\\Revisions\\Figures\\hm_diffgen.png')
    # #plt.show()
    # plt.clf()
    # plt.imshow(error_gen2, aspect='auto',cmap=cmap,vmin=0, vmax=0.6)
    # plt.xlabel("Diff w/NS")
    # plt.colorbar()
    # plt.savefig('.\\Evaluation\\Revisions\\Figures\\hm_diffns.png')
    # #plt.show()
    # plt.clf()


    # scores_gt_arr = np.array(scores_gt).reshape(-1, 5)
    # scores_gen_arr = np.array(scores_gen).reshape(-1, 5)
    # scores_gen2_arr = np.array(scores_gen2).reshape(-1, 5)

    # scores_gt_arr = norm_cols(scores_gt_arr)
    # scores_gen_arr = norm_cols(scores_gen_arr)
    # scores_gen2_arr = norm_cols(scores_gen2_arr)
    # # create_boxplot_triplets(scores_gt_arr ,scores_gen_arr, scores_gen2_arr,"all")
    # create_histogram(scores_gt.T, scores_gen2.T, "hey")


if __name__ == "__main__":


    # main()
    # perform_comparisons()

    values = [151, 207, 218]
    base = values[0]

    increments = [
    values[0],                # base segment
    values[1] - values[0],    # from 151 to 207
    values[2] - values[1]     # from 207 to 218
]

# percentages RELATIVE TO PREVIOUS VALUE
percentages = [0]  # base has no % change
for i in range(1, len(values)):
    pct = (values[i] / values[i-1] - 1) * 100
    percentages.append(pct)

colors = ["steelblue", "slategrey", "cadetblue"]

fig, ax = plt.subplots(figsize=(8, 2))

left = 0
for inc, pct, color in zip(increments, percentages, colors):
    ax.barh(0, inc, left=left, color=color)

    if inc > 0:
        if pct == 0:
            label = f"Base ({base} min)"
        else:
            label = f"+{pct:.1f}%"
        ax.text(left + inc/2, 0, label,
                ha="center", va="center", color="black", fontsize=10)  # ← black text
    left += inc

ax.set_xlim(0, 230)
ax.set_yticks([])
ax.set_xlabel("Minutes")
ax.set_title("Horizontal Stacked Bar Showing % Increase from Previous")
plt.show()

    
        


    
    


    
        
        

