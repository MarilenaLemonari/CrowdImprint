
# Structure
This is the main tool of the project. The inferred behaviour distributions are save in `".../Experiments/Inference"`, whereas histograms of simulator-specific predictions are saved in `".../Tool/Distributions"`.

# Tool
To utilise run `".../Tool/run_sl.bat"`. This performs the following steps (shown with example parameters):
```
python3 preprocess.py --scenario "Scenario1_friends" --source "CCP"
python3 run_sl.py --scenario "Scenario1_friends" --source "CCP" # in C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Experiments 
python3 run_similarity.py --scenario "Scenario1_friends" --source "CCP" --sim_agent "agent1.npz"
python3 build_distributions.py 
```