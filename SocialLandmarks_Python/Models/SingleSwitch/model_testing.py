from imports import *
from helper_functions import *
from data_loader_keras import *

# PERFECT DATA:
# cd C:\PROJECTS\SocialLandmarks
# Execute .\.venv\Scripts\activate 
# cd C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data
# python3 C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\SingleSwitch\model_testing.py


if __name__ ==  '__main__':
    trained_model = load_model("C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\SingleSwitch\model_test.h5")
    print("SUCCESS! Trained Model is Loaded.")

    # Train & Validation data aka "SEEN" data during training.
    # x_train, y_train =  load_data_keras()
    # print("SUCCESS! Seen Data is Loaded.")
    # seen_cm = make_cm(trained_model, x_train, y_train, "Seen Data")
    # print("Seen Data CM")
    # print(seen_cm)
    # results_train = trained_model.evaluate(x_train, y_train, batch_size=x_train.shape[0])
    # print(results_train)

    # Test data aka "UNSEEN" during training.
    x_test, y_test =  load_data_keras(test =  True)
    print("SUCCESS! Test Data is Loaded.")

    exit()
    index = 100
    x_test = x_test[:index]
    y_test = y_test[:index]

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