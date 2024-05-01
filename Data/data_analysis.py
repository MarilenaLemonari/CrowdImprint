# IMPORTS:
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from scipy.interpolate import interp1d

# INSTRUCTIONS:
#   cd C:\PROJECTS\SocialLandmarks
#   .venv/Scripts/activate
#   cd .\Data\
#   python3 .\data_analysis.py

def preprocess_data(source):
  # opentraj
  print("preprocess_data()")

  """
  source == "ETH":
    [frame_number pedestrian_ID pos_x pos_z pos_y v_x v_z v_y ]
    https://icu.ee.ethz.ch/research/datsets.html
    25fps videos BUT 2.5 fps annotations (0.4s timestep)
  """

  if source == "ETH":
    # traj_path =  "C:/PROJECTS/SocialLandmarks/Data/Trajectories/eth_hotel.txt"
    traj_dir = "C:/PROJECTS/SocialLandmarks/Data/Trajectories"
    all_files = os.listdir(traj_dir)
    eth_files = [file for file in all_files if file.endswith(".txt") and "eth" in file]
    agents_traj = []
    
    for i, eth_file in enumerate(eth_files):
      traj_path =  f"C:/PROJECTS/SocialLandmarks/Data/Trajectories/{eth_file}"

      traj_array = np.loadtxt(traj_path)

      column_names = ["Frame_No", "Pedestrian_ID", "pos_x", "pos_z", "pos_y", "v_x", "v_z", "v_y" ]
      df = pd.DataFrame(traj_array, columns = column_names)
      df = df.drop("pos_z", axis = 1)
      df = df.drop("v_z", axis = 1)
      df = df.drop("v_x", axis = 1)
      df = df.drop("v_y", axis = 1)
      num_agents = int(df["Pedestrian_ID"].max())
      agents_traj_list = []
      for agent_id in range(1,num_agents+1):

        agent_traj = df[df["Pedestrian_ID"] == agent_id]
        agent_traj = agent_traj.drop("Pedestrian_ID", axis = 1)
        agents_traj_values = agent_traj.values
        # Remove non existent agent IDs
        if len(agents_traj_values) == 0:
          continue
        frame_cutoff = agents_traj_values[-1,0] # TODO: Custom e.g.25
        start_frame = agents_traj_values[0,0]
        cutoff = 0
        cutoff_list = []
        start_list = [0]
        for i in range(1,agents_traj_values.shape[0]):
          end_frame = agents_traj_values[i,0]
          if end_frame-start_frame > frame_cutoff:
            cutoff += 1
            cutoff_list.append(i)
            start_list.append(i)
            start_frame = end_frame
        cutoff_list.append(agents_traj_values.shape[0]+1)

        for c in range(cutoff+1):
          agents_traj_list.append(agents_traj_values[start_list[c]:cutoff_list[c],:])

      agents_traj.append(agents_traj_list)

    return agents_traj

  else:
    print("INPUT ERROR: Source Type Not Supported!")
    return

def visualize_agent_traj(agents_traj, title):
  print("visualize_agent_traj()")

  max_dist = []
  max_width = []
  processed_points = []
  for i, agent_traj in enumerate(agents_traj[:10]):
    if len(agent_traj) <= 0:
      continue #TODO: pedestrian id 21
    
    if np.ndim(agent_traj) == 1:
      agent_traj = agent_traj.reshape((1,3))

    start_pos_x = agent_traj[0,1]
    start_pos_y = agent_traj[0,2]

    x_s = agent_traj[:,1]-start_pos_x
    y_s = agent_traj[:,2]-start_pos_y

    end_pos_x = x_s[-1]
    end_pos_y = y_s[-1]

    if abs(end_pos_y) < 0.001:
      continue #TODO: why stationary characters?
    
    tan_value = end_pos_x/end_pos_y

    theta = abs(np.arctan(tan_value))
    if end_pos_x > 0 and end_pos_y > 0:
      theta = (np.deg2rad(180) - theta)
    elif end_pos_x > 0 and end_pos_y <= 0:
      theta = theta
    elif end_pos_x <= 0 and end_pos_y > 0:
      theta = - (np.deg2rad(180) - theta)
    else:
      theta = - theta
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    points = np.column_stack((x_s, y_s))
    rotated_points = np.dot(points, rotation_matrix)
    processed_points.append(rotated_points)
    max_dist.append(rotated_points[-1,1])
    max_width.append(max(rotated_points[:,0]))

  max_value = max_dist[np.argmax(np.abs(max_dist))]
  max_value_w = max_width[np.argmax(np.abs(max_width))]
  # if abs(max_value) >= abs(max_value_w):
  #   max_overall = max_value
  # else:
  #   max_overall = max_value_w
  
  max_height = []
  data = []
  for i, points in enumerate(processed_points):   
    scale_factor = max_value/points[-1,1]
    y_stretched = [scale_factor * yi for yi in points[:,1]]
    max_height.append(max(np.abs(y_stretched)))
    datapoints = np.column_stack((points[:,0],np.array(y_stretched)))
    data.append(datapoints)
    plt.plot(points[:,0], y_stretched, 'firebrick')
  
  max_value_h = max_height[np.argmax(np.abs(max_height))]
  plt.plot(0,0,'k',marker='.', markersize=8)
  plt.plot(0,max_value,'k',marker='.', markersize=8)
  plt.title(f"{title}")
  tol_y = 1.5
  plt.ylim(-abs(max_value_h)-tol_y, abs(max_value_h)+tol_y)
  # plt.xlim(-abs(int(max_overall/4))-1, abs(int(max_overall/4))+1) #TODO: regulate
  tol_x = 0.1
  plt.xlim(-abs(max_value_w)-tol_x, abs(max_value_w)+tol_x) 
  plt.show()

  return data, [max_value, max_value_h, max_value_w]

def perform_dtw(data, max_list):

  print("perform_dtw()")

  # x = np.array([ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
  # 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
  # 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
  # 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
  # 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00,
  # 0.00000000e+00, -8.12579934e-03, -1.29995256e-02, -6.27880755e-02,
  # -4.60151318e-03,  1.80790258e-16] )
  # y = np.array([ 0. ,         0.,          0.,          0.  ,        0. ,         0.,
  # 0.      ,    0. ,         0. ,         0. ,         0. ,         0.,
  # 0. ,         0.  ,        0. ,         0. ,         0.  ,        0.,
  # 0.   ,       0.  ,        0. ,        -1.03871126, -1.66169846, -2.8771257,
  # -5.73785893, -7.72248985])
  # # y = np.array([ 0.00000000e+00, -8.12579934e-03, -1.29995256e-02, -6.27880755e-02,
  # # -4.60151318e-03,  1.80790258e-16] )
  # # x = np.array([ 0. ,        -1.03871126, -1.66169846, -2.8771257,
  # # -5.73785893, -7.72248985])
  # plt.plot(x,y)
  # indices = np.column_stack((0,np.where(x != 0)))
  # unique_y = x[indices[0,:]]
  # unique_x = y[indices[0,:]]
  # degree = 3
  # coefficients = np.polyfit(unique_x,unique_y, degree)
  # poly_func = np.poly1d(coefficients)
  # # interp_func = interp1d(np.unique(x), np.unique(y), kind = 'linear')
  # x_interp = np.linspace(unique_x.min(), unique_x.max(), 1000)
  # y_interp = poly_func(x_interp)
  # # y_interp = interp_func(x_interp)
  # plt.plot(y_interp,x_interp,'r.-')
  # plt.show()
  # exit()

  [max_value, max_value_h, max_value_w] = max_list

  n_points = []
  for i,datapoints in enumerate(data):
    n_points.append(datapoints.shape[0])
  max_length =  max(n_points)
 
  # Padd data:
  plt.figure(figsize=(8, 6))
  for i,datapoints in enumerate(data):
    datapoints = datapoints.reshape((1,datapoints.shape[0], datapoints.shape[1]))
    padded_data = np.pad(datapoints, ((0, 0), (max_length - n_points[i], 0), (0, 0)), mode='constant')
    plt.plot(padded_data[0,:,0], padded_data[0,:,1],'lightgrey',linestyle = 'dotted')
    if i == 0:
      data_full = padded_data
    else:
      data_full = np.vstack((data_full, padded_data))

  n_clusters = 3
  kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw")
  kmeans.fit(data_full)

  centroids = kmeans.cluster_centers_

  for centroid in centroids:
      x = centroid[:, 0]
      y = centroid[:, 1]
      plt.plot(x, y,'firebrick', linestyle = 'dashdot')
      indices = np.column_stack((0,np.where(x != 0)))
      unique_y = x[indices[0,:]]
      unique_x = y[indices[0,:]]
      degree = 3
      coefficients = np.polyfit(unique_x,unique_y, degree)
      poly_func = np.poly1d(coefficients)
      # interp_func = interp1d(np.unique(x), np.unique(y), kind = 'linear')
      x_interp = np.linspace(unique_x.min(), unique_x.max(), 1000)
      y_interp = poly_func(x_interp)
      # y_interp = interp_func(x_interp)
      plt.plot(y_interp,x_interp,'dimgrey', linewidth = 2)

  plt.plot(0,0,'orange',marker='.', markersize=10)
  plt.plot(0,max_value,'orange',marker='.', markersize=10)
  tol_y = 1.5
  plt.ylim(-abs(max_value_h)-tol_y, abs(max_value_h)+tol_y)
  tol_x = 0.1
  plt.xlim(-abs(max_value_w)-tol_x, abs(max_value_w)+tol_x) 
  plt.title('Representative Curves using DTW')
  plt.xlabel('X-axis')
  plt.ylabel('Y-axis')
  #plt.legend(["one","two","three"])
  plt.show()

if __name__ ==  '__main__':
  print("Main")
  agents_traj = preprocess_data(source = "ETH")
  data, [max_value, max_value_h, max_value_w] = visualize_agent_traj(agents_traj = agents_traj[0], title = "ETH_Hotel Trajectories")
  # visualize_agent_traj(agents_traj = agents_traj[1], title = "ETH_Road Trajectories")
  perform_dtw(data, [max_value, max_value_h, max_value_w])


