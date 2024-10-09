
# Structure
The experiments are split into INFERENCE and EVALUATION, in `"cdir =C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Experiments"`

## Evaluation
Evaluation is performed on the recorded instructed data (`"C:\PROJECTS\SocialLandmarks\Data\Tracking\YOLO\Tracker\Images\Instructed"`).
```
preprocess_existing.py
evaluation_existing.py
```
The outputs in `"cdir\Evaluation\GroundTruths"` are:
1. `"c_dict.json"` contains the prediction of InF combinations for each of the 125 inputs.
2. `"final_dict.json"`shows, for each input, the (i) ground truth class, (ii) predicted class, and (3) if it is a match.
3. `"confusion_matrix.png"`

## Inference
```
preprocess_existing.py 
inference_existing.py 
visualize_gen_traj.py
```