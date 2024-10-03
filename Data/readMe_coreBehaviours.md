
# Description
1. Go to directory `cdir = ".../SocialLandmarks/Data"`
2. `cdir/Trajectories` has the csv files of the existing datasets we used for trajectory analysis.
3. `cdir/Results` contains .png images of the set of the 5 (max) most commonly calculated curves for each dataset, along with the plot of all formatted trajectories and the DTW curves.
4.  `cdir/Results`  also contains the chosen core behaviours (presented as sketched Interaction Fields).
5. We use data from 4 different scenarios i.e., Flock, Hotel, Road, University Campus (2 batches of data), and Commercial Street (3 batches). 
6. `cdir/data_analysis.py` contains the python script used to generate the aforementioned png images.

# Execution
- Requirements of venv are found in `".../SocialLandmarks/requirements.txt"`
- Double click on `coreBehaviours.bat` or run `cdir/coreBehaviours.bat`

# Note
At this stage, the `cdir/Tracking` folder is not used.
