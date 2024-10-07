
# Description
The scripts in this directory generate the ground truth data, used for training, validation and testing of the main model. 
1. Go to directory `cdir = ".../SocialLandmarks/SocialLandmarks_Python/Data"`
2. `cdir/generate_trajectories.py` is the first script to run, and generates simulated trajectories subject to a sampled scenario, e.g., combination of InFs, etc.
3. `cdir/generate_trajectory_images.py` processed the csv trajectories, and generate the tif images (trajectory images).
4. `cdir/Trajectories` contains csv trajectories.
5. `cdir/Images` contains tif images.
6. `cdir/PythonFiles`  contains npz files, corresponding to the images. 

# Execution
- Requirements of venv are found in `".../SocialLandmarks/requirements.txt"`
- Also, here the [InFs](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.14491)+[UMANS](https://project.inria.fr/crowdscience/project/ocsr/umans/) framework is necessary to have in another repository, called `"InteractionFieldsUMANS"`.
- Double click on `generateTrainingData.bat` or run `cdir/generateTrainingData.bat`

# Note
The training data for each type i.e., trajectories, images, and python files, are contained in the `"SingleSwitch"` subfolder.