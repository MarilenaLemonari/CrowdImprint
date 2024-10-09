@echo off
cd C:\PROJECTS\SocialLandmarks\Data\Tracking\YOLO\Tracker
call myenv\Scripts\activate
python -u .\detection.py
pause