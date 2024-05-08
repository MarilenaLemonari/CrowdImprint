from inference import *

# Instructions:
# cd C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Experiments
# python3 inference_existing.py

if __name__ ==  '__main__':

    model_name = "model_test.h5"
    model_type = "keras"
    batch_size = 4
    x_test = np.random.random((batch_size, 32, 32, 1)) # torch.randn(batch_size, 1, 32, 32)
    y_test = np.ones(batch_size) # torch.ones(batch_size)

    predictions, predicted_labels = model_inference(model_name, model_type, x_test, y_test)
    print(predicted_labels)
    