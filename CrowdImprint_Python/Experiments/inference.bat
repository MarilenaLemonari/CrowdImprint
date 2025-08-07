@echo off
cd C:\PROJECTS\SocialLandmarks
call .venv\Scripts\activate
cd C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Experiments
python3 -u inference_existing.py
python3 -u visualize_gen_traj.py
pause