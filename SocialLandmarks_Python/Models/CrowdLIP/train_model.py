from imports import *
from data_loader import *
# from CNNKeras import *
# from config import *

os.environ['WANDB_API_KEY']="29162836c3095b286c169bf889de71652ed3201b"

# TODO:
# cd C:\PROJECTS\SocialLandmarks
# Execute .\.venv\Scripts\activate 
# go to cd C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data
# Execute python3 C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\CrowdLIP\train_model.py

if __name__ ==  '__main__':
    """
    ...
    """

    images, gt  = load_data_keras()
    print("SUCCESS! Data Loaded. Details: ", images.shape, len(gt))

    exit()
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
    model.save("C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\SingleSwitch\model_test.h5")
    print("SUCCESS! Model is Saved.")


# import tensorflow as tf
# from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, GlobalAveragePooling2D, Lambda
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# import tensorflow.keras.backend as K
# import numpy as np

# # Define constants
# IMG_SHAPE = (224, 224, 3)  # Image input shape
# EMBED_DIM = 128            # Embedding dimension for images

# # Image encoder model
# def build_image_encoder():
#     img_input = Input(shape=IMG_SHAPE)
#     x = Conv2D(32, (3, 3), activation='relu')(img_input)
#     x = Conv2D(64, (3, 3), activation='relu')(x)
#     x = GlobalAveragePooling2D()(x)
#     x = Dense(EMBED_DIM, activation='relu')(x)
#     model = Model(img_input, x)
#     return model

# # Contrastive loss function
# def contrastive_loss(y_true, y_pred):
#     margin = 1.0
#     y_true = tf.cast(y_true, y_pred.dtype)
#     square_pred = K.square(y_pred)
#     margin_square = K.square(K.maximum(margin - y_pred, 0))
#     return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

# # Euclidean distance layer
# def euclidean_distance(vectors):
#     x, y = vectors
#     sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
#     return K.sqrt(K.maximum(sum_square, K.epsilon()))

# # Build the image encoder
# image_encoder = build_image_encoder()

# # Define inputs
# img_a = Input(shape=IMG_SHAPE)
# img_b = Input(shape=IMG_SHAPE)

# # Process both images through the same encoder
# emb_a = image_encoder(img_a)
# emb_b = image_encoder(img_b)

# # Calculate the Euclidean distance between the embeddings
# distance = Lambda(euclidean_distance)([emb_a, emb_b])

# # Final model
# model = Model([img_a, img_b], distance)

# # Compile the model
# model.compile(optimizer=Adam(), loss=contrastive_loss)

# # Generate some dummy data
# num_samples = 1000
# dummy_images_a = np.random.random((num_samples, *IMG_SHAPE)).astype(np.float32)
# dummy_images_b = np.random.random((num_samples, *IMG_SHAPE)).astype(np.float32)
# dummy_labels = np.random.randint(0, 2, size=(num_samples, 1)).astype(np.float32)  # 1 if same source, 0 otherwise

# # Train the model
# model.fit([dummy_images_a, dummy_images_b], dummy_labels, batch_size=32, epochs=10)

# # Function to compute similarity score
# def compute_similarity(img1, img2):
#     img1 = np.expand_dims(img1, axis=0)
#     img2 = np.expand_dims(img2, axis=0)
#     return model.predict([img1, img2])[0]

# # Example usage
# similarity_score = compute_similarity(dummy_images_a[0], dummy_images_b[0])
# print(f'Similarity score: {similarity_score}')
