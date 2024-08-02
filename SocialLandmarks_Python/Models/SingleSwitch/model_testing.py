from imports import *
from helper_functions import *
from data_loader_keras import *
from data_loader import *
from CNNPytorch import *
from train_pytorch_model import *
from config import *

# PERFECT DATA:
# cd C:\PROJECTS\SocialLandmarks
# Execute .\.venv\Scripts\activate 
# cd C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data
# python3 C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\SingleSwitch\model_testing.py

def test_keras():
    trained_model = load_model("C:\\PROJECTS\\SocialLandmarks\\SocialLandmarks_Python\\Models\\SingleSwitch\\model_final.pth")
    print("SUCCESS! Trained Model is Loaded.")

    # Train & Validation data aka "SEEN" data during training.
    x_train, y_train =  load_data_keras()
    print("SUCCESS! Seen Data is Loaded.")
    seen_cm = make_cm(trained_model, x_train, y_train, "Seen Data")
    print("Seen Data CM")
    print(seen_cm)
    results_train = trained_model.evaluate(x_train, y_train, batch_size=x_train.shape[0])
    print(results_train)

    # Test data aka "UNSEEN" during training.
    x_test, y_test =  load_data_keras(test =  True)
    print("SUCCESS! Test Data is Loaded.")
    test_cm  = make_cm(trained_model, x_test, y_test, "Test")
    print("Test CM:")
    print(test_cm)
    results_test = trained_model.evaluate(x_test, y_test, batch_size=x_test.shape[0])
    print(results_test)

    # Performance overall:
    test_metrics = {
    "performance_seen": results_train,
    "rperformance_test": results_test
    }
    filename = 'performance_metrics.json'
    with open(filename, 'r') as file:
        data = json.load(file)
    data.update(test_metrics)
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print(f"Data successfully written to {filename}")


if __name__ ==  '__main__':
    trained_model, criterion, optimizer, device = instantiate_model()
    trained_model.load_state_dict(torch.load("C:\\PROJECTS\\SocialLandmarks\\SocialLandmarks_Python\\Models\\SingleSwitch\\model_final.pth"))
    print("SUCCESS! Trained Model is Loaded.")

    # Train & Validation data aka "SEEN" data during training.
    images, gt =  load_data()
    print("SUCCESS! Seen Data is Loaded.")
    seendataloader = setup_config_test(images, gt)
    results_train = validate(trained_model, seendataloader, criterion, device, CM = True, name = "Seen")
    print(results_train)

    # Test data aka "UNSEEN" during training.
    x_test, y_test =  load_data(test =  True)
    print("SUCCESS! Test Data is Loaded.")
    testloader = setup_config_test(x_test, y_test)
    results_test = validate(trained_model, testloader, criterion, device, CM = True, name = "Test")
    print(results_test)

    # Performance overall:
    test_metrics = {
    "performance_seen": results_train,
    "performance_test": results_test
    }
    filename = 'performance_metrics.json'
    with open(filename, 'r') as file:
        data = json.load(file)
    data.update(test_metrics)
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print(f"Data successfully written to {filename}")