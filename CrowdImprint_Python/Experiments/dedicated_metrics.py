from imports import *
import glob

if __name__ == "__main__":
    
    dict_path = ".\\Evaluation\\Revisions"
    files = glob.glob(f"{dict_path}/*.json")

    metrics = np.zeros((len(files), 5, 2))

    for i,file in enumerate(files):
        name = file.split(dict_path)[1].split(".json")[0].split("final_dict_")[1]

        stop_m = []
        circle_m = []
        attract_m = []
        uni_m = []
        avoid_m = []
        with open(file, "r") as f:
            metrics_dict = json.load(f)
        for ag, ag_metrics in metrics_dict.items():
            stop_m.append(ag_metrics["stop_metric"])
            circle_m.append(ag_metrics["circling_metric"])
            attract_m.append(ag_metrics["attract_metric"])
            uni_m.append(ag_metrics["uni_metric"])
            avoid_m.append(ag_metrics["avoid_metric"])
        
        metrics[i,0,:] = np.array([np.mean(stop_m), np.std(stop_m)])
        metrics[i,1,:] = np.array([np.mean(circle_m), np.std(circle_m)])
        metrics[i,2,:] = np.array([np.mean(attract_m), np.std(attract_m)])
        metrics[i,3,:] = np.array([np.mean(uni_m), np.std(uni_m)])
        metrics[i,4,:] = np.array([np.mean(avoid_m), np.std(avoid_m)])
    
        print(name, " : ")
        print(metrics[i,:,:])
        
        
        
        

