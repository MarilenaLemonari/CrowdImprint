from imports import *
from data_loader import *
from SiameseModel import *
from config import *

os.environ['WANDB_API_KEY']="29162836c3095b286c169bf889de71652ed3201b"

# TODO:
# cd C:\PROJECTS\SocialLandmarks
# Execute .\.venv\Scripts\activate 
# go to cd C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data
# Execute python3 C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\CrowdLIP\train_model.py

if __name__ ==  '__main__':
    """
    Images should be of shape (32,32,1)
    """

    x_train_a, x_train_b, y_train = load_data(n_reps  = 5000)
    x_val_a, x_val_b, y_val = load_data(n_reps  = 5000, val=True)
#     images_a, images_b, gt = load_data(n_reps  = 5000)
#     print("SUCCESS! Data Loaded. Details: ", images_a.shape, images_b.shape, len(gt))
    model = instantiate_model()
    print("SUCCESS! Model is Instantiated.")

#     x_train_a, x_train_b, y_train, x_val_a, x_val_b,  y_val, config = setup_config_keras(images_a, images_b, gt)  
    x_train_a, x_train_b, y_train, x_val_a, x_val_b,  y_val, config = setup_config_keras(x_train_a, x_val_a, x_train_b, x_val_b, y_train, y_val)
    print( x_train_a.shape, x_train_b.shape, y_train.shape, x_val_a.shape, x_val_b.shape,  y_val.shape)

    model.fit([x_train_a, x_train_b], y_train, epochs=config.epochs, batch_size=config.batch_size,
            validation_data=([x_val_a, x_val_b], y_val),
            callbacks=[WandbCallback()])
    wandb.finish()

    # Saving:
    # # Example usage
    # similarity_score = compute_similarity(dummy_images_a[0], dummy_images_b[0])
    # print(f'Similarity score: {similarity_score}')
    model.save("C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\CrowdLIP\model_CLIP.h5")
    print("SUCCESS! Model is Saved.")





