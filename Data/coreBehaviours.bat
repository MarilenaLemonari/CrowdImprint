@echo off
cd C:\PROJECTS\SocialLandmarks
call .venv\Scripts\activate
cd .\Data\
echo Virtual Environment activated!
python3 -u .\data_analysis.py
pause