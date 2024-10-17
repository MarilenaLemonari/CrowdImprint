@echo off
cd C:\PROJECTS\SocialLandmarks
call .venv\Scripts\activate
cd C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Tool
python3 -u preprocess.py --scenario "Scenario1_friends" --source "CCP"
cd C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Experiments 
python3 -u run_sl.py --scenario "Scenario1_friends" --source "CCP"
cd C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Tool 
python3 -u run_similarity.py --scenario "Scenario1_friends" --source "CCP" --sim_agent "agent1.npz"
python3 -u build_distributions.py 
pause