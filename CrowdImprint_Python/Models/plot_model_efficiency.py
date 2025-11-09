import numpy as np
import matplotlib.pyplot as plt
import json

def main():
    json_path = ".\\model_efficiency.json"
    with open(json_path, 'r') as f:
        model_data = json.load(f)
    
    x = [0,1,2]
    for i, (key, value) in enumerate(model_data.items()):
        c_data = [value["noswitch"], value["cim"], value["doubleswitch"]]
        if i == 0:
            t_traj = c_data
        elif i== 1:
            t_img = c_data
        elif i == 2:
            t_train = c_data
        elif i == 3:
            t_inf = c_data

    plt.plot(x,t_traj, "-.", color = "slategrey")
    plt.plot(x,t_traj, ".", color = "slategrey", markersize = 10)
    #
    plt.plot(x,t_img, color = "#88A6CE")
    plt.plot(x,t_img, ".", color = "#88A6CE", markersize = 10)
    #
    plt.plot(x,t_train, color = "firebrick")
    plt.plot(x,t_train, ".", color = "firebrick", markersize = 10)
    #
    plt.plot(x,t_inf, color = "#DCDCDC")
    plt.plot(x,t_inf, ".", color = "#DCDCDC", markersize = 10)
    #
    plt.title("Computational Efficiency")
    plt.show()

if __name__ == "__main__":
    
    main()
    
