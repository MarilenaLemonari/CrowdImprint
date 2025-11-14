from imports import *


def visualize_agent_traj(agents_traj, title, plot = True,source=np.array([0.09,0.6])):
    print("visualize_agent_traj()")

    max_dist = []
    max_width = []
    processed_points = []
    processed_source = []
    num_traj = len(agents_traj)

    for i, agent_traj in enumerate(agents_traj):
        if len(agent_traj) <= 0:
            continue #TODO: pedestrian id 21
        
        if np.ndim(agent_traj) == 1:
            agent_traj = agent_traj.reshape((1,3))

        # plt.plot(agent_traj[:,1], agent_traj[:,2], 'slategrey')
        # plt.title("Raw Trajectories")

        start_pos_x = agent_traj[0,1]
        start_pos_y = agent_traj[0,2]


        x_s = agent_traj[:,1]-start_pos_x
        y_s = agent_traj[:,2]-start_pos_y

        source_x = source[0]
        source_z = source[1]
        source_x -= start_pos_x
        source_z -= start_pos_y
        # plt.plot(x_s, y_s, 'slategrey')
        # plt.plot([0], [0],"o" ,c='black')
        # plt.plot([source_x], [source_z], "o", c="cadetblue")
        # plt.xlim([-1,1])
        # plt.ylim([-2,2])
        # plt.show()
        # exit()

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
        source_points = np.column_stack((source_x,source_z))
        rotated_points = np.dot(points, rotation_matrix)
        rotated_source = np.dot(source_points, rotation_matrix)
        processed_points.append(rotated_points)
        processed_source.append(rotated_source)
        max_dist.append(rotated_points[-1,1])
        max_width.append(max(rotated_points[:,0]))

        # plt.plot(rotated_points[:,0], rotated_points[:,1], 'slategrey')
        # plt.plot([0], [0],"o" ,c='black')
        # plt.plot(rotated_source[:,0], rotated_source[:,1], "o", c="cadetblue")
        # plt.xlim([-1,1])
        # plt.ylim([-2,2])
        # plt.show()
        # plt.clf()
        # exit()

    max_value = max_dist[np.argmax(np.abs(max_dist))]
    max_value_w = max_width[np.argmax(np.abs(max_width))]
    # if abs(max_value) >= abs(max_value_w):
    #   max_overall = max_value
    # else:
    #   max_overall = max_value_w
    

    max_height = []
    data = []
    datasource = []
    for i, points in enumerate(processed_points):   
        scale_factor = max_value/points[-1,1]
        y_stretched = [scale_factor * yi for yi in points[:,1]]
        source_z_stretched = [scale_factor * zi for zi in processed_source[i][:,1]]
        max_height.append(max(np.abs(y_stretched)))
        datapoints = np.column_stack((points[:,0],np.array(y_stretched)))
        sourcepoints = np.column_stack((processed_source[i][:,0], source_z_stretched))
        data.append(datapoints)
        datasource.append(source_points)

        plt.plot(points[:,0], y_stretched, 'slategrey')
        # plt.plot([0], [0],"o" ,c='black')
        plt.plot(processed_source[i][:,0], source_z_stretched, "o", c="cadetblue")
        # plt.xlim([-1,1])
        # plt.ylim([-2,2])
        # plt.show()
        # exit()
    
    max_value_h = max_height[np.argmax(np.abs(max_height))]
    plt.plot(0,0,'k',marker='.', markersize=8)
    plt.plot(0,max_value,'k',marker='.', markersize=8)
    plt.title(f"{title} (total of {num_traj} trajectories)")
    tol_y = 1.5
    # plt.ylim(-abs(max_value_h)-tol_y, abs(max_value_h)+tol_y)
    # plt.xlim(-abs(int(max_overall/4))-1, abs(int(max_overall/4))+1) #TODO: regulate
    tol_x = 0.1
    # plt.xlim(-abs(max_value_w)-tol_x, abs(max_value_w)+tol_x) 
    # plt.xlim([-1,1])
    plt.ylim([-5,5])
    # plt.show()
    # exit()

    if plot == True:
        plt.savefig(f".\\Figures\\{title}.png")
        plt.show()

    return data, datasource, [max_value, max_value_h, max_value_w]

def perform_dtw(data, datasource, max_list, n_clusters, degree):
  #print(datasource)
  arr = np.array(datasource)[:,0,:]
  print(arr.shape)

  source_x = arr[:,0]
  source_z = arr[:,1]
  print(source_x.shape, source_z.shape)

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
      # plt.scatter(x, y, s = 2, color='firebrick',label = f'centroid_{counter}')
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

  # plt.plot(0,0,'orange',marker='.', markersize=10, label = 'point_spawn')
  # plt.plot(0,max_value,'orange',marker='.', markersize=10, label = 'point_goal')

  for line in plt.gca().lines:
    line.set_visible(False)
    if 'point' in line._label:
      line.set_visible(True)

  for line in plt.gca().lines: 
    if 'if_' in line._label:
       line.set_visible(True)
       if_part, num_part = line._label.split('num_')
       plt.plot(source_x, source_z, 'o', c="cadetblue")
       plt.ylim([-5,5])
       plt.xlim([-2,2])
       plt.title(f'No.Trajectories: {num_part}')
       plt.savefig(f'.\\Figures\\{if_part}')
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
  plt.savefig('.\\Figures\\Representative Curves using DTW')
  plt.show()

# ========================== 

def extract_pseudo_ground_trajectories(df, img_w=1280, img_h=800, invert_z=True):
    """
    Extract per-visitor trajectories as (frameid, x, z) from bounding boxes.

    Args:
        df: pandas DataFrame with columns:
            ['visitorid', 'cameraid', 'frameid',
             'bb_x', 'bb_y', 'bb_width', 'bb_height', ...]
        img_w: image width in pixels (default 1280)
        img_h: image height in pixels (default 800)
        invert_z: if True, z = 1 - cy_norm so that larger z means "further up" in image

    Returns:
        dict of dicts:
        {
          visitorid: {
              cameraid: [(frameid, x, z), (frameid, x, z), ...]  # sorted by frameid
          },
          ...
        }
    """
    # Bottom center of bounding box in image space
    df = df.copy()
    df["cx"] = df["bb_x"] + df["bb_width"] / 2.0
    df["cy"] = df["bb_y"] + df["bb_height"]

    # normalize to [0,1]
    df["x_norm"] = df["cx"] / float(img_w)
    df["z_norm"] = df["cy"] / float(img_h)
    if invert_z:
        df["z_norm"] = 1.0 - df["z_norm"]

    # sort for consistent trajectories
    df_sorted = df.sort_values(["visitorid", "cameraid", "frameid"])

    trajectories = {}
    group_info = {}

    for (vid, cam), group in df_sorted.groupby(["visitorid", "cameraid"]):


        # make sure this visitor exists
        if vid not in trajectories:
            trajectories[vid] = {}

        traj_list = []
        visitor_group = []
        for _, row in group.iterrows():
            visitor_group.append(row["groupid"])
            
            frame = int(row["frameid"])
            x = float(row["x_norm"])
            z = float(row["z_norm"])
            traj_list.append((frame, x, z))

        trajectories[vid][cam] = traj_list
        group_info[f"{vid}_{cam}"] = visitor_group

    return trajectories, group_info

def load_data():
    path = "annotations.dat"

    # Column names in the order you described
    cols = [
        "visitorid",
        "cameraid",
        "frameid",
        "bb_x", "bb_y", "bb_width", "bb_height",
        "bbV_x", "bbV_y", "bbV_width", "bbV_height",  # last name duplicated in description
        "gazeAngle_x", "gazeAngle_y",
        "filename",
        "operaid",
        "groupid"
    ]

    df = pd.read_csv(
        path,
        sep=",",
        header=None,   # assuming the file has no header row
        names=cols
    )
    return df

def plot_paths(arr, name ,source=np.array([0.09,0.6])):

    init = arr[0,:] 

    plt.plot(arr[:,0], arr[:,1], '.-',c="slategrey")
    plt.plot(arr[0,0], arr[0,1], 'o', c="black")
    # plt.plot(arr[frames.index(3011),1], arr[frames.index(3011),2], 'x')
    plt.plot(source[0], source[1], 'o', c="firebrick")
    plt.xlim(-0.1,1.1)
    plt.ylim(-0.1,1.1)
    plt.xlabel("x")
    plt.ylabel("z")
    plt.savefig(f".\\Trajectories\\{name}.png")
    plt.close()
    #plt.show()

def preprocess_traj(traj, plot_bool=False, stretch_vertical = 10/8):

    for visitor_id, traj_dict in traj.items():
        for camera_id, path_cam in traj[visitor_id].items():
            path_array = np.array(path_cam)
            # maybe here convert first column to timeframe?
            path_array[:,2] *=  1#stretch_vertical
            path_cam_v2 = list(path_array)
            traj[visitor_id][camera_id] = path_cam_v2
            name = f"{visitor_id}_{camera_id}"
            if (camera_id == 1) and (plot_bool == True):
                plot_paths(path_array[:,1:], name)

def split_groups(traj, group_info):

    group_traj = {}

    for visitor_id, traj_dict in traj.items():
        for camera_id, path_cam in traj[visitor_id].items():
            path_array = np.array(path_cam)
            group_key = f"{visitor_id}_{camera_id}"
            lst = group_info[group_key]
            
            indices = [i for i in range(1, len(lst)) if lst[i] != lst[i-1]]
            indices += [len(lst)]
            i0 = 0
            for i in indices:
                groupid = lst[i0]
                new_key = f"{visitor_id}_{camera_id}_{groupid}"
                new_arr = path_array[i0:i, :]
                group_traj[new_key] = new_arr
                i0 = i
                # plot_paths(new_arr[:,1:], new_key)

    return group_traj

def format_data(group_traj):

    agent_traj = []
    for ag, ag_traj in group_traj.items():
        agent_traj.append(ag_traj)

    return [agent_traj]

def main():

    df = load_data()

    traj, group_info = extract_pseudo_ground_trajectories(df)

    # Process data:
    # preprocess_traj(traj, plot_bool=False)
    group_traj = split_groups(traj, group_info)
    agents_traj = format_data(group_traj)

    data, datasource,[max_value, max_value_h, max_value_w] = visualize_agent_traj(agents_traj = agents_traj[0], title = "museum_visitors")
    perform_dtw(data, datasource, [max_value, max_value_h, max_value_w], n_clusters = 5, degree = 3)
    

if __name__ == "__main__":

    main()

    # '''
    # @InProceedings\{BLSKD15,
    #     author = “Bartoli, Federico and Lisanti, Giuseppe and Seidenari,
    #     Lorenzo and Karaman, Svebor and Del Bimbo, Alberto”,
    #     title = “MuseumVisitors: a dataset for pedestrian and group detection,
    #     gaze estimation and behavior understanding”,
    #     booktitle = “Proc. of CVPR Int’l. Workshop on Int. Workshop on Group
    #     And Crowd Behavior Analysis And Understanding”,
    #     year = “2015”,
    #     url = “http://www.micc.unifi.it/publications/2015/BLSKD15”
    # }
    # '''