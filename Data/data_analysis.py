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

def skiprows(index):
    if index < 0:
        return True
    else:
        return (index % 10 != 0)

def preprocess_data(source):
  # opentraj
  print("preprocess_data()")

  """
  source == "ETH":
    [frame_number pedestrian_ID pos_x pos_z pos_y v_x v_z v_y ]
    https://icu.ee.ethz.ch/research/datsets.html
    25fps videos BUT 2.5 fps annotations (0.4s timestep)
  
  source == "Flock"
    ./Flock/*.csv 
    [time; pos_x; pos_y]
    0.04s timestep => 25fps
  """

  if source == "ETH":
    """
    returns: list 'agents_traj'
            list of #eth files i.e., eth_hotel and eth_road,
            within which a list of #agents,
            of positional arrays of shape (#frames, 3) with columns: [frame_no, pos_x, pos_y].
    """
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

  elif source == "Flock":
    """
    returns: list 'agents_traj'
            list of flock scenarios i.e., 1,
            within which a list of #agents,
            of positional arrays of shape (#timesteps, 3) with columns: [timestep, pos_x, pos_y]
    """

    traj_dir = "C:/PROJECTS/SocialLandmarks/Data/Trajectories/Flock/"
    all_agent_files = os.listdir(traj_dir)
    column_names = ["Timestep", "pos_x", "pos_y" ]

    agents_traj = []
    agents_traj_list = []
    timestep = 0.04

    for agent_id,agent_file in enumerate(all_agent_files):
      # Load file:
      agent_traj = pd.read_csv(traj_dir+agent_file, header=None)
      agent_traj[column_names] = agent_traj[0].str.split(';', expand=True)
      agent_traj[column_names[0]] = agent_traj[column_names[0]].astype(float) 
      agent_traj[column_names[1]] = agent_traj[column_names[1]].astype(float)
      agent_traj[column_names[2]] = agent_traj[column_names[2]].astype(float)
      agent_traj.drop(0, axis=1, inplace=True)
      agents_traj_values = agent_traj.values
      
      # Cutoff trajectories according to specified framerate/duration:
      frame_cutoff = int(agents_traj_values[-1,0]/timestep + 1) # TODO: Custom e.g.25
      start_frame = int(agents_traj_values[0,0]/timestep + 1)
      cutoff = 0
      cutoff_list = []
      start_list = [0]
      for i in range(1,agents_traj_values.shape[0]):
        end_frame = int(agents_traj_values[i,0]/timestep + 1)
        if end_frame-start_frame > frame_cutoff:
          cutoff += 1
          cutoff_list.append(i)
          start_list.append(i)
          start_frame = end_frame
      cutoff_list.append(agents_traj_values.shape[0]+1)

      for c in range(cutoff+1):
        agents_traj_list.append(agents_traj_values[start_list[c]:cutoff_list[c],:])
      # agents_traj_list.append(agents_traj_values) # *(386,3)

    agents_traj.append(agents_traj_list)
    return agents_traj

  elif source == "Zara":
    """
    returns: list 'agents_traj'
          list of #zara files i.e., zara1, zara2, zara3,
          within which a list of #agents,
          of positional arrays of shape (#frames, 5) with columns: [frame_no, pos_x, pos_y, pixel_x, pixel_y].
          2.5fps
    """
  
    traj_dir = "C:/PROJECTS/SocialLandmarks/Data/Trajectories/Zara/"
    zara_folders = [folder for folder in os.listdir(traj_dir) if os.path.isdir(os.path.join(traj_dir, folder))]
    agents_traj = []
    timestep = 0.4
    column_names = ["Timestep", "pos_x", "pos_y", "pixel_x", "pixel_y"]

    for i, zara_folder in enumerate(zara_folders):
      all_agent_files = os.listdir(traj_dir+zara_folder)
      agents_traj_list = []
      if zara_folder == "Zara03":
        column_names = ["Timestep", "pos_x", "pos_y"]

      # Load agent trajectory:
      for agent_id,agent_file in enumerate(all_agent_files):
        agent_traj = pd.read_csv(traj_dir+zara_folder+"/"+agent_file, header=None, skiprows= skiprows)
        agent_traj[column_names] = agent_traj[0].str.split(';', expand=True)
        agent_traj[column_names[0]] = agent_traj[column_names[0]].astype(float) 
        agent_traj[column_names[1]] = agent_traj[column_names[1]].astype(float)
        agent_traj[column_names[2]] = agent_traj[column_names[2]].astype(float)
        agent_traj.drop(0, axis=1, inplace=True)
        if zara_folder != "Zara03":
          agent_traj = agent_traj.drop("pixel_x", axis = 1)
          agent_traj = agent_traj.drop("pixel_y", axis = 1)
        agents_traj_values = agent_traj.values

        # Cutoff trajectories according to specified framerate/duration:
        frame_cutoff = int(agents_traj_values[-1,0]/timestep + 1) # TODO: Custom e.g.25
        start_frame = int(agents_traj_values[0,0]/timestep + 1)
        cutoff = 0
        cutoff_list = []
        start_list = [0]
        for i in range(1,agents_traj_values.shape[0]):
          end_frame = int(agents_traj_values[i,0]/timestep + 1)
          if end_frame-start_frame > frame_cutoff:
            cutoff += 1
            cutoff_list.append(i)
            start_list.append(i)
            start_frame = end_frame
        cutoff_list.append(agents_traj_values.shape[0]+1)

        for c in range(cutoff+1):
          agents_traj_list.append(agents_traj_values[start_list[c]:cutoff_list[c],:])
        # agents_traj_list.append(agents_traj_values)
      
      agents_traj.append(agents_traj_list)

    return agents_traj
    

  else:
    print("INPUT ERROR: Source Type Not Supported!")
    return

def visualize_agent_traj(agents_traj, title, plot = True):
  print("visualize_agent_traj()")

  max_dist = []
  max_width = []
  processed_points = []
  num_traj = len(agents_traj)

  for i, agent_traj in enumerate(agents_traj):
    if len(agent_traj) <= 0:
      continue #TODO: pedestrian id 21
    
    if np.ndim(agent_traj) == 1:
      agent_traj = agent_traj.reshape((1,3))

    # plt.plot(agent_traj[:,1], agent_traj[:,2], 'slategrey')

    start_pos_x = agent_traj[0,1]
    start_pos_y = agent_traj[0,2]


    x_s = agent_traj[:,1]-start_pos_x
    y_s = agent_traj[:,2]-start_pos_y
    # plt.plot(x_s, y_s, 'slategrey')

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
    plt.plot(points[:,0], y_stretched, 'slategrey')
  
  max_value_h = max_height[np.argmax(np.abs(max_height))]
  plt.plot(0,0,'k',marker='.', markersize=8)
  plt.plot(0,max_value,'k',marker='.', markersize=8)
  plt.title(f"{title} (total of {num_traj} trajectories)")
  tol_y = 1.5
  # plt.ylim(-abs(max_value_h)-tol_y, abs(max_value_h)+tol_y)
  # plt.xlim(-abs(int(max_overall/4))-1, abs(int(max_overall/4))+1) #TODO: regulate
  tol_x = 0.1
  # plt.xlim(-abs(max_value_w)-tol_x, abs(max_value_w)+tol_x) 

  if plot == True:
    plt.savefig(f"{title}")
    plt.show()

  return data, [max_value, max_value_h, max_value_w]

def perform_dtw(data, max_list, n_clusters, degree):

  print("perform_dtw()")

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
    plt.plot(padded_data[0,:,0], padded_data[0,:,1],'lightgrey',linestyle = 'dotted', label = f'gt_{i}')
    if i == 0:
      data_full = padded_data
    else:
      data_full = np.vstack((data_full, padded_data))

  # data_full_x = data_full[:,:,0]
  # data_full_y = data_full[:,:,1]
  # data_full[:,:,0] = data_full_y
  # data_full[:,:,1] = data_full_x
  # n_clusters = 5
  kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw")
  kmeans.fit(data_full)

  centroids = kmeans.cluster_centers_

  # Get amount of data in each cluster
  cluster_labels = kmeans.labels_
  unique_labels, counts = np.unique(cluster_labels, return_counts=True)
  dict = {}
  for label, count in zip(unique_labels, counts):
      dict[label] = count
      print(f"Cluster {label}: {count} examples")

  counter = 0
  for centroid in centroids:
      num = dict[counter]
      counter += 1
      x = centroid[:, 0]
      y = centroid[:, 1]
      plt.plot(x, y, linestyle  = 'dashdot', markersize=2, color='firebrick',label = f'centroid_{counter}')
      indices = np.column_stack((0,np.where(x != 0)))
      unique_y = x[indices[0,:]]
      unique_x = y[indices[0,:]]
      # degree = 3
      coefficients = np.polyfit(unique_x,unique_y, degree)
      poly_func = np.poly1d(coefficients)
      # interp_func = interp1d(np.unique(x), np.unique(y), kind = 'linear')
      x_interp = np.linspace(unique_x.min(), unique_x.max(), 1000)
      y_interp = poly_func(x_interp)
      # y_interp = interp_func(x_interp)
      plt.plot(y_interp,x_interp,'dimgrey', linewidth = 2, label = f'if_{counter}num_{num}')

  plt.plot(0,0,'orange',marker='.', markersize=10, label = 'point_spawn')
  plt.plot(0,max_value,'orange',marker='.', markersize=10, label = 'point_goal')

  for line in plt.gca().lines:
    line.set_visible(False)
    if 'point' in line._label:
      line.set_visible(True)

  for line in plt.gca().lines: 
    if 'if_' in line._label:
       line.set_visible(True)
       if_part, num_part = line._label.split('num_')
       plt.title(f'No.Trajectories: {num_part}')
       plt.savefig(f'{if_part}')
       line.set_visible(False)

  for line in plt.gca().lines:
    line.set_visible(True)

  tol_y = 1.5
  # plt.ylim(-abs(max_value_h)-tol_y, abs(max_value_h)+tol_y) # TODO: regulate
  tol_x = 0.1
  # plt.xlim(-abs(max_value_w)-tol_x, abs(max_value_w)+tol_x) 
  plt.title('Representative Curves using DTW')
  plt.xlabel('X-axis')
  plt.ylabel('Y-axis')
  #plt.legend(["one","two","three"])
  plt.savefig('Representative Curves using DTW')
  plt.show()

if __name__ ==  '__main__':
  print("Main")
  # agents_traj = preprocess_data(source = "ETH")
  # agents_traj = preprocess_data(source = "Flock")
  agents_traj = preprocess_data(source = "Zara")

  # data, [max_value, max_value_h, max_value_w] = visualize_agent_traj(agents_traj = agents_traj[0], title = "ETH_Hotel Trajectories")
  # data, [max_value, max_value_h, max_value_w] = visualize_agent_traj(agents_traj = agents_traj[1], title = "ETH_Road Trajectories")
  # data, [max_value, max_value_h, max_value_w] = visualize_agent_traj(agents_traj = agents_traj[0], title = "Flock Trajectories")
  # data, [max_value, max_value_h, max_value_w] = visualize_agent_traj(agents_traj = agents_traj[0], title = "Zara_1 Trajectories")
  data, [max_value, max_value_h, max_value_w] = visualize_agent_traj(agents_traj = agents_traj[1], title = "Zara_2 Trajectories")
  
  perform_dtw(data, [max_value, max_value_h, max_value_w], n_clusters = 5, degree = 3)