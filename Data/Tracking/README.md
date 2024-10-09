
# Description
1. The two subfolders in this directory are the two ways that have been attempted to track videos, i.e., YOLO and TRex. We end up only using YOLO.
2. Go to directory `cdir = ".../SocialLandmarks/Data/Tracking/YOLO/Tracker"`
3. `"cdir/calibration.py"` performs the camera calibration according to our custom data capturing, based on checkboard images.
4. `"cdir/detection.py` generates the tracked trajectories from input videos. 
5. `"cdir/generate_images_from_recordings.py"` generates the trajectory images from the tracked data.
6. `"cdir/Calibration"` should be used for the checkboard images (for calibration).
7. `"cdir/Trajectories"` is where the tracke data is saved.
8. `"cdir/Images"` is where the generated trajectory images (compatible with our model) are saved.
9. `"cdir/yolov10n.pt"` is the utilised YOLO model version.

# Execution
- To avoid conficts, we make a new venv `"cdir/myenv"` with requirements `"cdir/req.txt"`.
- Navigate to directory `".../SocialLandmarks/Data/Tracking"`.
- Double click  or run `trackingVideos.bat`, provided you have already calibrated the camera.
- You can edit the `.bat` to input the new camera matrix `(3,3)` and distortion coefficients `(5,)` using `--camera_matrix` and `--dist_coeffs` in the python line. 
- Otherwise, the dafault (ours) values will be used.
```
@echo off
cd C:\PROJECTS\SocialLandmarks\Data\Tracking\YOLO\Tracker
call myenv\Scripts\activate
python -u .\detection.py --camera_matrix e.g., '1,0,0,0,1,0,0,0,1'
pause
```

# Note
Scripts `cdir/vosualize_recorded_trajectories.py` and `cdir/detection2people.py` are supplmentary, and mostly used for experimentation.
