@echo off
cd C:\PROJECTS\SocialLandmarks
call .venv\Scripts\activate
cd C:\PROJECTS\DataDrivenInteractionFields\InteractionFieldsUMANS\examples
echo Virtual Environment activated!
python3 -u C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\generate_trajectories.py
cd C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data
python3 .\generate_trajectory_images.py
pause