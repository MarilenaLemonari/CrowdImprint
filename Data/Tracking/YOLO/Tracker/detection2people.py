import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import os
import csv
from scipy.signal import savgol_filter

# Initialize the YOLO model
model = YOLO("yolov10n.pt")

# Camera matrix and distortion coefficients
camera_matrix = np.array([
    [4.73908241e+03, 0.00000000e+00, 2.89321683e+03],
    [0.00000000e+00, 4.73819501e+03, 2.02846582e+03],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])
dist_coeffs = np.array([1.10624914e-02, -2.59719368e+00, -5.70461677e-03, -1.22012466e-02, 6.42489045e+00])

# Video directory and files
video_dir = 'E:\WorkCYENS\DataRecording\Videos_Instructed'
video_files = glob.glob(os.path.join(video_dir, '*.mp4'))

video_files = ["E:\WorkCYENS\DataRecording_OG\Videos_Scenarios\scenario1_friends_subject1.mp4"]

person1_trajectories = []
person2_trajectories = []

# Process each video file
for video_file in video_files:
    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Trajectories for both people
    person1_traj = []
    person2_traj = []

    # Progress bar
    with tqdm(total=total_frames, desc=f"Processing {os.path.basename(video_file)}", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Undistort the frame
            h, w = frame.shape[:2]
            new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
            undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_mtx)

            # YOLO detection
            results = model(undistorted_frame)

            
            person_count = 0  # To track how many people are detected
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].numpy()  # Bounding box coordinates
                    confidence = box.conf[0].item()  # Confidence score
                    class_id = box.cls[0].item()  # Class ID

                    if confidence > 0.5 and int(class_id) == 0:  # Class 0 is 'person' in COCO dataset
                        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                        person_count += 1

                        # Get the bottom center of the bounding box (feet position)
                        feet_pixel = np.array([x + w / 2, y + h])

                        # Convert the bottom center of the bounding box to world coordinates
                        feet_world_coords = cv2.undistortPoints(np.expand_dims(feet_pixel, axis=0), camera_matrix, dist_coeffs, P=new_camera_mtx)
                        feet_world_coords = feet_world_coords[0][0] / 100  # Scale down to meters

                        # Store the world coordinates of the feet
                        if person_count == 1:
                            person1_traj.append(feet_world_coords)
                        elif person_count == 2:
                            person2_traj.append(feet_world_coords)

                        # Stop after detecting two people
                        if person_count == 2:
                            break

            # Update the progress bar
            pbar.update(1)

    cap.release()
    cv2.destroyAllWindows()

    person1_trajectories.append(np.array(person1_traj))
    person2_trajectories.append(np.array(person2_traj))

timestep = 1/30
source_x = 7.25
source_y = 5.14
c = 0
for traj in person2_trajectories:
    time =  [i * timestep for i in range(len(traj))]
    x = traj[:,0]
    y = traj[:,1]
    source = [0] * len(traj)
    source[0] = source_x
    source[1] = source_y
    # Stretch
    for i in range(len(y)):
        y_i = y[i]
        if y_i >= source_y:
            y_i = (y_i - source_y) * (1+abs(y_i-source_y)) + source_y
        else:
            y_i = (y_i - source_y) * (2) + source_y
        y[i] = y_i
    # output_name = "class_" + video_files[c].split("class_")[1].split('.mp4')[0] + ".csv"
    output_name = video_files[c].split(".mp4")[0].split("Videos_Scenarios\\")[1] + ".csv"
    output_file = f'.\Trajectories\PersonTrajectories\{output_name}'
    c += 1
    # Smoothen
    window_size = 5
    poly_order = 2
    smoothed_y = savgol_filter(y, window_size, poly_order)
    # Iterate
    window_s = 3
    smoothed_y = smoothed_y[::window_s]
    smoothed_x = x[::window_s]
    time = time[::window_s]
    source = source[:len(time)]

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # writer.writerow(['Time', 'Pos_x', 'Pos_y', 'Source'])
        for i in range(len(smoothed_x)):
            writer.writerow([time[i], smoothed_x[i], smoothed_y[i], source[i]])