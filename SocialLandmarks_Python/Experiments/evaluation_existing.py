from inference import *
from inference_existing import *
from preprocess_existing import *
from generate_custom_trajectories import *
import os
from tqdm import tqdm

# Instructions:
# cd C:\PROJECTS\SocialLandmarks
# .\.venv\Scripts\activate
# cd C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Experiments
# python3 evaluation_existing.py

def generate_guided_trajectory(beh_combination, start_pts, or_pts, source_pt, end_time, timestep, name):
    start_x, start_z = start_pts
    or_x = or_pts[0] - start_x
    or_z = or_pts[1] - start_z

    os.chdir("C:\PROJECTS\DataDrivenInteractionFields\InteractionFieldsUMANS\examples")
    behavior_list = ["Unidirectional_Down","Attractive_Multidirectional","Other_CircleAround", "AvoidNew", "MoveTF", "Stop"]

    dictionary = {}
    for i in range(len(behavior_list)-1):
        dictionary[i] = behavior_list[i]

    
    init_positions = np.array([[source_pt,source_pt],[start_x,start_z]])
    build_xml(init_positions, [0], dictionary, end_time, delta_time = timestep)


    string = beh_combination.strip("[]'")
    InF1, InF2 = map(int, string.split('_'))
    field_1 = InF1
    field_2 = InF2

    weight = np.zeros((1,len(behavior_list)-1))
    actionTimes = np.ones((1,len(behavior_list)-1))*(-1)
    inactiveTimes = np.ones((1,len(behavior_list)-1))*(-1)
    T = random.randint(2,int(end_time-2)) # TODO: Find most optimal?

    if field_1 != 5 and field_2 != 5:
        weight[0,field_1] = 1
        weight[0,field_2] = 1

        inactiveTimes[0,field_1] = T
        actionTimes[0,field_2] = T

        inactiveTimes[0,field_2] = end_time
        actionTimes[0,field_1] = 0
    elif field_1 == 5 and field_2 != 5:
        weight[0,field_2] = 1
        inactiveTimes[0,field_2] = end_time
        actionTimes[0,field_2] = T
    elif field_1 != 5 and field_2 == 5:
        weight[0,field_1] = 1
        inactiveTimes[0,field_1] = T
        actionTimes[0,field_1] = 0

    generate_instance(init_positions,weight,actionTimes,inactiveTimes,or_x, or_z,dictionary, groupID = 0)
    
    # Save trajectories
    mode = "Evaluation\\Flock"
    S_true=make_trajectory(1,mode,name = name)

    return S_true

def metric_unidirectional(trajectory):
    """
    A measure correcpodning to the "unidirectional" movement.
        1. Every 1s take spawn and goal positions.
        2. Create the straight line between them.
        3. Measure similarity/deviation from this line.
    """
    return trajectory

def metric_attractive(trajectory):
    """
    A measure corresponding to the "attract" movement.
    Basically, this is a combination of the unidirectional metric plus a measure of the distance to the source.
    So here we calculate the distance to the source over time.
    """
    return trajectory

def metric_circle(trajectory):
    """
    A measure of the "circling around" movement.
    This corresponds to the amount of time that the agents maintains a standard distance to the source.
    """
    return trajectory

def metric_avoid(trajectory):
    """
    A measure of the "avoid" movement. 
        1. Evey second take the spawn and goal positions.
        2. Fit a second degree polynomial curve and mesure similarity/deviation.
    """
    return trajectory

def metric_stop(trajectory):
    """
    A measure of the "stopping" behaviour.
    This corresponds to the amount of time that speed is 0 (or less than a tolerance).
    """
    return trajectory

if __name__ ==  '__main__':
    """
    Evaluation with existing dataset is limited to measuring % success in behavior replication.
    The following code then does the following:
        i. 
        iii. Trajectory comparison with inputs/gts given same initial positions.
        WE assume appropriate source placement.
    """

    model_name = "trial2.pth"
    model_type = "pytorch"
    dataset_name = "Flock"
    # dataset_name = "Zara"
    # dataset_name = "Students"

    folder_path = f'C:\\PROJECTS\\SocialLandmarks\\SocialLandmarks_Python\\Data\\PythonFiles\\{dataset_name}\\'  
    x_test, pred_dict = load_inference_data(folder_path, return_dict=True)
    c_batch_size = x_test.shape[0]
    batch_size = 32
    predictions, predicted_labels = model_inference(model_name, model_type, x_test, batch_size)
    combinations, c_dict = decode_labels(predicted_labels, pred_dict)
    print(c_dict)