
# Structure
The experiments are split into INFERENCE and EVALUATION, in `"cdir =C:\PROJECTS\CrowdImprint\CrowdImprint_Python\Experiments"`

## Evaluation
Evaluation is performed on the recorded instructed data (`"C:\PROJECTS\CrowdImprint\Data\Tracking\YOLO\Tracker\Images\Instructed"`).
```
preprocess_existing.py
evaluation_existing.py
```
The outputs in `"cdir\Evaluation\GroundTruths"` are:
1. `"c_dict.json"` contains the prediction of InF combinations for each of the 125 inputs.
2. `"final_dict.json"`shows, for each input, the (i) ground truth class, (ii) predicted class, and (3) if it is a match.
3. `"confusion_matrix.png"`

## Inference
- To use the framework, you have to have preprocessed trajectories (`.npz`) of agents interacting to a certain type of source e.g., ATM machine.
- Run `"inference.py"` specifying the number of agents you want the new simulation to have (`default  = 3`).
- Then, you can visualize the generated trajectories `"visualize_gen_traj.py"`.
So, overall:
```
preprocess_existing.py 
inference_existing.py 
visualize_gen_traj.py
```
The outputs are:
1. In `"...\CrowdImprint\CrowdImprint_Python\Experiments\Inference"`, where the distribution of the predicted behaviour combinations can be found.
2. In `"...\CrowdImprint\CrowdImprint_Python\Data\Trajectories\Inference\{input_name}"`, where the new `.csv` trajectories can be found, along with their plot and corresponding predicted combinations. 

### Note
- Given you have preprocessed data, you can perform inference via `"inference.bat"`.
- If you only want to obtain the inferred distribution of behaviour without novel generations, run `"run_sl.py"`.