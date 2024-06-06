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
        Model inputs are (1,32,32) #TODO verify
    Outputs:
        Model outputs are predictions of 1st and 2nd InF.
        The code saves the trained model,
        and the confusion matrices. 
    """
    start_time = time.time()
    # x_train, y_train  = load_data_keras()
    # x_val, y_val  = load_data_keras(val = True)
    images, gt  = load_data_keras()
    x_train, x_val, y_train, y_val = train_test_split(images, gt, test_size=0.2, random_state=42)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("SUCCESS! Data Loaded. Details: ", x_train.shape, y_train.shape, x_val.shape, y_val.shape)
    print(f"Loading Time: {elapsed_time:.2f} seconds")

    model = instantiate_model()
    print("SUCCESS! Model is Instantiated.")
    config, datagen , reduce_lr = setup_config_keras()


    start_time = time.time()
    model.fit(x_train, y_train, epochs=config.epochs, batch_size=config.batch_size,
            validation_data=(x_val, y_val),
            callbacks=[WandbCallback()])
    #     model.fit( datagen.flow(x_train, y_train, batch_size=config.batch_size),  epochs=config.epochs,
            # validation_data=(x_val, y_val),
            # callbacks=[WandbCallback(), reduce_lr])
    wandb.finish()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training Time: {elapsed_time:.2f} seconds")

    # Saving:
    t_cm = make_cm(model, x_train, y_train, "Training")
    v_cm  = make_cm(model, x_val, y_val, "Validation")
    print("Training CM")
    print(t_cm)
    print("Validation CM:")
    print(v_cm)

    results_train = model.evaluate(x_train, y_train, batch_size=x_train.shape[0])
    results_val = model.evaluate(x_val, y_val, batch_size=x_val.shape[0])
    performance_metrics = {
    "time": elapsed_time,
    "results_train": results_train,
    "results_val": results_val
    }
    filename = 'performance_metrics.json'
    with open(filename, 'w') as json_file:
        json.dump(performance_metrics, json_file, indent=4)
    model.save("C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\SingleSwitch\model_test.h5")
    print("SUCCESS! Model is Saved.")