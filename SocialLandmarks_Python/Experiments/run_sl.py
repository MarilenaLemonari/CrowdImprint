from imports import *
from inference import *
from inference_existing import *

if __name__ ==  '__main__':
    parser = argparse.ArgumentParser(description="Process command-line inputs.")
    parser.add_argument('--scenario', type=str, default="Scenario3_exhibit", help='Specific Scenario to perform Inference.')
    parser.add_argument('--source', type=str, default="SL", help='Synthetic Trajectory Source.')
    args = parser.parse_args()

    # Instructions:
    # cd C:\PROJECTS\SocialLandmarks
    # .\.venv\Scripts\activate
    # cd C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Experiments
    # python3 run_sl.py

    folder_dir = "C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\PythonFiles\CaseStudy"
    # name = "\\RecordedData\\Scenario5_foodcourt"
    # name = "\\SL\\Scenario2_guard"
    # name = "\\CCP\\Scenario3_exhibit"
    name = f"\\{args.source}\\{args.scenario}"
    folder_path = folder_dir + name

    x_tool = load_inference_data(folder_path)
    c_batch_size = x_tool.shape[0]
    batch_size = 32

    model_name = "model_final.pth"
    model_type = "pytorch"

    predictions, predicted_labels = model_inference(model_name, model_type, x_tool, batch_size)
    combinations, c_dict = decode_labels(predicted_labels)
    print(c_dict)

    file_path = f"{folder_path}\\predictions.json"
    with open(file_path, "w") as json_file:
        json.dump(c_dict, json_file)


    