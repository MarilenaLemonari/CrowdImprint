import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import os
import csv

# python3
# cd C:\PROJECTS\SocialLandmarks\Data\Tracking\YOLO\Tracker
# myenv\Scripts\activate
# python detection.py


model = YOLO("yolov10n.pt")  # TODO

# Load the camera matrix and distortion coefficients (replace with your actual values)
camera_matrix = np.array([
    [4.73908241e+03, 0.00000000e+00, 2.89321683e+03],
    [0.00000000e+00, 4.73819501e+03, 2.02846582e+03],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])
dist_coeffs = np.array([1.10624914e-02, -2.59719368e+00, -5.70461677e-03, -1.22012466e-02, 6.42489045e+00])
video_dir = './Videos/'  
video_files = glob.glob(os.path.join(video_dir, '*.mp4'))

all_feet_trajectories = []

for video_file in video_files:
    cap = cv2.VideoCapture(video_file)
    
    # Get the total number of frames in the video for the progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # List to store the world coordinates of the feet for the current video
    feet_trajectories = []

    # Process the video frames with a progress bar
    with tqdm(total=total_frames, desc=f"Processing {os.path.basename(video_file)}", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Undistort the frame
            h, w = frame.shape[:2]
            new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
            undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_mtx)

            # Perform YOLO object detection
            results = model(undistorted_frame)

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].numpy()  # Bounding box coordinates
                    confidence = box.conf[0].item()  # Confidence score
                    class_id = box.cls[0].item()  # Class ID

                    if confidence > 0.5 and int(class_id) == 0:  # Class 0 is 'person' in COCO dataset
                        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                        
                        # Draw the bounding box
                        cv2.rectangle(undistorted_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(undistorted_frame, f"Person: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Get the bottom center of the bounding box
                        feet_pixel = np.array([x + w / 2, y + h])
                        
                        # Convert the bottom center of the bounding box to world coordinates
                        feet_world_coords = cv2.undistortPoints(np.expand_dims(feet_pixel, axis=0), camera_matrix, dist_coeffs, P=new_camera_mtx)
                        feet_world_coords = feet_world_coords[0][0] / 100 
                        
                        # Store the world coordinates of the feet
                        feet_trajectories.append(feet_world_coords)

            # Display the frame with bounding boxes
            # cv2.imshow('Frame with Bounding Boxes', undistorted_frame) # TODO
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            # Update the progress bar
            pbar.update(1)

    cap.release()
    cv2.destroyAllWindows()

    # Convert the list of trajectories to a NumPy array for easier handling
    feet_trajectories = np.array(feet_trajectories)

    # Add to the list of all trajectories
    all_feet_trajectories.append(feet_trajectories)

timestep = 1/30
source_x = 7.25
source_y = 5.14
c = 0
for traj in all_feet_trajectories:
    time =  [i * timestep for i in range(len(traj))]
    x = traj[:,0]
    y = traj[:,1]
    source = [0] * len(traj)
    source[0] = source_x
    source[1] = source_y
    y = (y - source_y) * 2 + source_y
    output_name = "class_" + video_files[c].split("class_")[1].split('.mp4')[0] + ".csv"
    output_file = f'.\Trajectories\{output_name}'
    c += 1
    window_size = 5
    smoothed_y = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
    smoothed_x = x[(window_size//2):-(window_size//2)]
    time = time[(window_size//2):-(window_size//2)]
    source = source[:len(time)]

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # writer.writerow(['Time', 'Pos_x', 'Pos_y', 'Source'])
        for i in range(len(smoothed_x)):
            writer.writerow([time[i], smoothed_x[i], smoothed_y[i], source[i]])

# # Plot the trajectories using matplotlib
# plt.figure(figsize=(10, 6))
# plt.plot(source_x, source_y, '*', markersize = 5)
# for traj in all_feet_trajectories:
#     plt.plot(traj[:, 0], traj[:, 1], 'o-', markersize=3)
# plt.xlabel('X (m)')
# plt.ylabel('Y (m)')
# plt.title('Feet Trajectories in World Space')
# plt.grid(True)
# # plt.xlim([0, 20])
# # plt.ylim([0, 20])
# plt.show()
