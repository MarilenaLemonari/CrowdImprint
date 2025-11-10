from imports import *
import glob

def get_decoder():
    decoder = {0:"1_1", 1:"1_2",2: "1_3", 3: "1_4", 4: "1_5",
                5: "2_1", 6: "2_2", 7: "2_3", 8: "2_4", 9: "2_5",
                10 :"3_1", 11: "3_2",12: "3_3", 13: "3_4", 14: "3_5",
                15: "4_1", 16: "4_2", 17: "4_3", 18: "4_4", 19: "4_5",
                20: "5_1", 21: "5_2", 22: "5_3", 23: "5_4", 24: "5_5"}
    return decoder

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
            
            if ("objectscenarios" in name.lower()) or ("persontrajectories" in name.lower()):
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

            elif scenario_prefix in name.lower():     
                
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

def create_spec_boxplots(gt, gen, title, direction, subplot_names):
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
    plt.savefig(f".\\Evaluation\\Revisions\\Boxplots\\PerMetric\\{title}.png")
    # plt.show()
    plt.clf()
    plt.close()


def create_boxplot_pairs(gt, gen, title,subplot_names=["stop_metric", "circle_metric", "attract_metric","uni_metric","avoid_metric"]):
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
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["COLL.", "GEN."], loc="upper right")
    ax.set_title("Collected vs Generated")

    plt.tight_layout()
    plt.savefig(f".\\Evaluation\\Revisions\\Boxplots\\{title}.png")
    plt.clf()
    plt.close()
    # plt.show()

def get_metrics_dict():
    name_dict = {"stop": 0, "circle":1, "attract":2, "continue":3, "uni":3, "avoid":4}
    return name_dict

def get_metric_class():
    metric_class = {"stop":5,"circle":3, "attract":2, "continue":1, "uni":1, "avoid":4}
    return metric_class

if __name__ == "__main__":
    
    name = "instructed" #instructed, scenarios
    direction = "avoid"

    if name == "scenarios":
        pairs_dict = get_scenario_pairs()
    else:
        pairs_dict = get_pairs_dict()
    

    if direction == "all":
        for key, metrics_value in pairs_dict.items():

            gt = metrics_value["gt"]
            gen = metrics_value["gen"] # col is metrics

            create_boxplot_pairs(gt, gen, title=name+"_"+key)

    else:
        if name == "scenarios":
            print("ERROR: Cannot do metric-specific boxplot for scenario.")
            exit()

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
    


    
        
        

