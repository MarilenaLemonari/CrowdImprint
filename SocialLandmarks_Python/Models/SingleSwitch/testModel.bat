@echo off
cd C:\PROJECTS\SocialLandmarks
call .venv\Scripts\activate
cd C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data

:: Prompt user for input
set /p userInput="Do you need to preprocess your input data? (yes/no): "

:: Check if the user input is "yes"
if /i "%userInput%"=="yes" (
    echo Running the additional script...
    python3 C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\SingleSwitch\preprocess_model_data.py
)

:: Now run the model testing script
python3 -u C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\SingleSwitch\model_testing.py
pause
