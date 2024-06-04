from imports import *
from data_loader_keras import *
from CNNKeras import *
from config import *

os.environ['WANDB_API_KEY']="29162836c3095b286c169bf889de71652ed3201b"

# TODO:
# cd C:\PROJECTS\SocialLandmarks
# Execute .\.venv\Scripts\activate 
# go to cd C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data
# Execute python3 C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\SingleSwitch\train_keras_model.py

if __name__ ==  '__main__':
    """
    Train Pytorch model to predict combinations of InF.
    Preprocessing:
        Load the npy files and build gt (x,y) pairs.
    Inputs:
        Model inputs are (1,32,32) 
    Outputs:
        Model outputs are predictions of 1st and 2nd InF.
        The code saves the trained model,
        and the confusion matrices. 
    """

    images, gt  = load_data_keras()
    print("SUCCESS! Data Loaded. Details: ", images.shape, len(gt))
    model = instantiate_model()
    print("SUCCESS! Model is Instantiated.")

    x_train, y_train, x_val, y_val, config = setup_config_keras(images, gt)

    model.fit(x_train, y_train, epochs=config.epochs, batch_size=config.batch_size,
            validation_data=(x_val, y_val),
            callbacks=[WandbCallback()])
    wandb.finish()

    # Saving:
    t_cm = make_cm(model, x_train, y_train, "Training")
    v_cm  = make_cm(model, x_val, y_val, "Validation")
    print("Training CM")
    print(t_cm)
    print("Validation CM:")
    print(v_cm)
    # cmatrices = {
    # "training_cm": t_cm,
    # "validation_cm": v_cm 
    # }
    # filename = 'cmatrices.json'
    # with open(filename, 'w') as json_file:
    #     json.dump(cmatrices, json_file, indent=4)
    model.save("C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\SingleSwitch\model_full.h5")
    print("SUCCESS! Model is Saved.")