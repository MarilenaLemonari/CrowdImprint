
# Description
The scripts in this directory to train and test the main model. 
1. Go to directory `cdir = ".../SocialLandmarks/SocialLandmarks_Python/Data"`
2. `C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\SingleSwitch\train_pytorch_model.py` is the main script,responsible fro the training. 
3. `C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\SingleSwitch\preprocess_model_data.py` transforms the images to python files that are used as inputs.
4. These python files are in `cdir/PythonFiles/SingleSwitch`.
5. `C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\SingleSwitch\model_testing.py` tests model on unseen data.
6. The model is built in `CNNPytorch.py`.
7. Training/Validation/Test Confusion Matrices along with the performance metrics (.json) are outputed in the `cdir` when running the program again; the old metric are saved in `cdir\Figures`.

# Execution
- Requirements of venv are found in `".../SocialLandmarks/requirements.txt"`
- For utilisation of this framework, re-training is not necessary.
- To test the model (and obtain Confusion Matrices for train/val/test), run testModel.bat. You will be prompted to choose if you need to preprocess the images, or not i.e., they are already in npz files.
```
cd C:\PROJECTS\SocialLandmarks
.venv\Scripts\activate
cd C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data

# if preprocessing needed:
python3 C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\SingleSwitch\preprocess_model_data.py
# else:
python3 -u C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\SingleSwitch\model_testing.py
```

# Note
- The remaining scripts are complementary to the three executible scripts mentioned above. 
- The current version of the training script has integrated wandb logs; modifications to avoid that are possible.  