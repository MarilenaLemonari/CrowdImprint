from imports import *
from SiameseModel import *
from helper_functions import *
from data_loader import *

# TODO:
# cd C:\PROJECTS\SocialLandmarks
# Execute .\.venv\Scripts\activate 
# go to cd C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data
# Execute python3 C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\CrowdLIP\test_model.py

def contrastive_loss(y_true, distance, margin=1.0):
    # distance = y_pred
    loss = tf.reduce_mean((1 - y_true) * tf.square(distance) +
                          y_true * tf.square(tf.maximum(margin - distance, 0)))
    return loss

if __name__ == "__main__":
    # Load Model:
    image_encoder = tf.keras.models.load_model("C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\CrowdLIP\image_encoder_model.h5")
    print("SUCCESS! Encoder is loaded.")

    # Load data:
    images_a, images_b, gt = load_data(40, test = True) # test = True
    gt_dict = {1: "MATCH", 0: "NO MATCH"}
    print(images_a.shape, images_b.shape, gt.shape)
    print("SUCCESS! Data is loaded.")
    
    # Usage:
    emb_a = image_encoder(images_a)
    emb_b = image_encoder(images_b)
    emb_a_norm = Lambda(normalize)(emb_a)
    emb_b_norm = Lambda(normalize)(emb_b)
    print(emb_b_norm.shape, emb_a_norm.shape)
    distance = Lambda(euclidean_distance)([emb_a_norm, emb_b_norm]) #bounded from 0 to 2
    distance_norm = distance/2
    print(distance_norm.shape)
    accuracy_v = 0
    total = 0
    for i in tqdm(range(gt.shape[0])):
        gt_pnt = gt_dict[gt[i]]
        distance_pnt = distance_norm[i,0]
        if distance_pnt.numpy() > 0.5:
            pred_value = 0
            pred = "MATCH"
        else:
            pred_value = 1
            pred = "NO MATCH"
        total += 1
        if int(pred_value) == int(gt[i]):
            accuracy_v += 1
        loss = contrastive_loss(gt[i], distance_pnt)
        # print("Distance for point ",i," is ", distance_pnt.numpy()," and gt is ", gt_pnt, " and LOSS: ", loss.numpy())
        #print("Prediction for point ",i," is ", pred," and gt is ", gt_pnt, " and LOSS: ", loss.numpy())

    print(accuracy_v, " of ", total, " correct => ", (accuracy_v/total) * 100, "%")
        

