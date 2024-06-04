from imports import *
from data_loader import *
from Conv1dModel import *
from config import *

os.environ['WANDB_API_KEY']="29162836c3095b286c169bf889de71652ed3201b"

# TODO:
# cd C:\PROJECTS\SocialLandmarks
# Execute .\.venv\Scripts\activate 
# go to cd C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data
# Execute python3 C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\Conv1d\train_traj_model.py

if __name__ ==  '__main__':
    """
    Conv1d.
    """

    X, labels, seq_len  = load_traj_data()
    print("SUCCESS! Data Loaded. Details: ", X.shape, labels.shape)
    model = instantiate_model(seq_len)
    print("SUCCESS! Model is Instantiated.")

    X_train, X_val, y_train, y_val, config = setup_config_conv1d(X, labels)  
    print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

    model.fit(X_train, y_train, epochs=config.epochs, batch_size=config.batch_size, validation_data=(X_val, y_val),
          callbacks=[WandbCallback()])
    wandb.finish()

    # Saving:
    # # Example usage
    # eval_results = model.evaluate(X_val, y_val)
    # print(f"Validation Loss: {eval_results[0]}")
    # print(f"Validation Accuracy: {eval_results[1]}")
    t_cm = make_cm(model, X_train, y_train, "Training")
    v_cm  = make_cm(model, X_val, y_val, "Validation")
    print("Training CM")
    print(t_cm)
    print("Validation CM:")
    print(v_cm)
    model.save("C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\Conv1d\model_test.h5")
    print("SUCCESS! Model is Saved.")





