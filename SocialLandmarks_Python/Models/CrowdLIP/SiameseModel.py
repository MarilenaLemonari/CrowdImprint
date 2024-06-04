from imports import *
from helper_functions import *

# python3 C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\CrowdLIP\SiameseModel.py

def build_image_encoder(IMG_SHAPE, EMBED_DIM):
    img_input = Input(shape=IMG_SHAPE)
    x = Conv2D(32, (3, 3), activation='relu')(img_input)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(EMBED_DIM, activation='relu')(x)
    model = Model(img_input, x)
    return model

def instantiate_model():
    IMG_SHAPE = (32, 32, 1)  # Image input shape
    EMBED_DIM = 128            # Embedding dimension for images
    
    image_encoder = build_image_encoder(IMG_SHAPE, EMBED_DIM)

    # Define inputs
    img_a = Input(shape=IMG_SHAPE)
    img_b = Input(shape=IMG_SHAPE)

    # Process both images through the same encoder
    emb_a = image_encoder(img_a)
    emb_b = image_encoder(img_b)

    # Calculate the Euclidean distance between the embeddings
    distance = Lambda(euclidean_distance)([emb_a, emb_b])

    model = Model([img_a, img_b], distance)
    model.compile(optimizer=Adam(), loss=contrastive_loss, metrics = [accuracy])

    return model

if __name__ ==  '__main__':
    model = instantiate_model()

    # # Generate some dummy data
    # num_samples = 1000
    # IMG_SHAPE = (32, 32, 1)
    # dummy_images_a = np.random.random((num_samples, *IMG_SHAPE)).astype(np.float32)
    # dummy_images_b = np.random.random((num_samples, *IMG_SHAPE)).astype(np.float32)
    # dummy_labels = np.random.randint(0, 2, size=(num_samples, 1)).astype(np.float32)  
    # print(model.predict([dummy_images_a, dummy_images_b]).shape)