from imports import *
from SiameseModel import *

def load_data_keras(folder_path):  

    loaded_data = np.load(folder_path)
    array_keys = loaded_data.files
    array_key = array_keys[0]
    array = loaded_data[array_key]
    if (array.dtype != 'float32'):
        print(folder_path)
        exit()
    image = [array]

    image = scale_to_standard_normal(image)
        
    return image

if __name__ ==  '__main__':
    npz_real = "C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\PythonFiles\ActedScenarios\Scenario1_friends\scenario1_friends_subject1.npz"
    npz_sim = "C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\PythonFiles\ActedScenarios\Scenario1_friends\scenario1_friends_subject1.npz"

    # Load Model
    image_encoder = tf.keras.models.load_model("C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\CrowdLIP\image_encoder_model.h5")
    print("SUCCESS! Encoder is loaded.")

    # Load Data
    real = load_data_keras(npz_real) 
    sim = load_data_keras(npz_sim) 
    print(real.shape, sim.shape)
    print("SUCCESS! Data is loaded.")

    # Usage:
    emb_a = image_encoder(real)
    emb_b = image_encoder(sim)
    emb_a_norm = Lambda(normalize)(emb_a)
    emb_b_norm = Lambda(normalize)(emb_b)
    print(emb_b_norm.shape, emb_a_norm.shape)
    distance = Lambda(euclidean_distance)([emb_a_norm, emb_b_norm]) #bounded from 0 to 2
    distance_norm = distance/2
    print(distance_norm.shape)
    score = 1 - distance_norm[0,0]
    simScore = f"{float(score)*100}%"
    print("Similarity Score is: ",float(score)*100, "%")
    # if distance_pnt.numpy() > 0.5:
    #     pred_value = 0
    #     pred = "MATCH"
    # else:
    #     pred_value = 1
    #     pred = "NO MATCH"

    folder_path =  os.path.dirname(npz_sim)
    file_path = f"{folder_path}\\similarityScores.json"
    with open(file_path, "w") as json_file:
        json.dump(simScore, json_file)
