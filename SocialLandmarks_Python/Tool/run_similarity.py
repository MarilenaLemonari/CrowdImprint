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
    parser = argparse.ArgumentParser(description="Process command-line inputs.")
    parser.add_argument('--scenario', type=str, default="Scenario3_exhibit", help='Specific Scenario to perform Inference.')
    parser.add_argument('--source', type=str, default="SL", help='Synthetic Trajectory Source.')
    parser.add_argument('--sim_agent', type=str, default="agent_1.npz", help='Simulated agent name.')
    args = parser.parse_args()

    scenario_l = (args.scenario).lower()
    real_dir = f"C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\PythonFiles\CaseStudy\RecordedData\{args.scenario}\\"
    reals = [f"{scenario_l}_subject1.npz", f"{scenario_l}_subject2.npz", f"{scenario_l}_subject3.npz",
             f"{scenario_l}_subject4.npz", f"{scenario_l}_subject5.npz"]
    sim_name = args.sim_agent
    full_dict = {}
    score_dict = {}
    scores = []
    for real_name in reals:
        npz_real = real_dir + real_name
        npz_sim = f"C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\PythonFiles\CaseStudy\\{args.source}\{args.scenario}\\" + sim_name
        # npz_sim = f"C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data\PythonFiles\CaseStudy\\{args.source}\Scenario2_guard\\" + sim_name

        # Load Model
        image_encoder = tf.keras.models.load_model("C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\CrowdLIP\image_encoder_model.h5")
        print("SUCCESS! Encoder is loaded.")

        # Load Data
        real = load_data_keras(npz_real) 
        sim = load_data_keras(npz_sim) 
        # print(real.shape, sim.shape)
        print("SUCCESS! Data is loaded.")

        # Usage:
        emb_a = image_encoder(real)
        emb_b = image_encoder(sim)
        emb_a_norm = Lambda(normalize)(emb_a)
        emb_b_norm = Lambda(normalize)(emb_b)
        # print(emb_b_norm.shape, emb_a_norm.shape)
        distance = Lambda(euclidean_distance)([emb_a_norm, emb_b_norm]) #bounded from 0 to 2
        distance_norm = distance/2
        # print(distance_norm.shape)
        score = 1 - distance_norm[0,0]
        scores.append(score)
        simScore = f"{float(score)*100}%"
        # print("Similarity Score is: ",float(score)*100, "%")
        # if distance_pnt.numpy() > 0.5:
        #     pred_value = 0
        #     pred = "MATCH"
        # else:
        #     pred_value = 
        #     pred = "NO MATCH"
        score_dict[real_name] = simScore

    score_dict["avg"] = str(float(sum(scores)/len(reals))*100) +"%"
    score_dict["max"] = str(float(max(scores))*100)+"%"
    print("Max. Similarity Score is: ",score_dict["max"])
    full_dict[sim_name] = score_dict
    folder_path =  os.path.dirname(npz_sim)
    file_path = f"{folder_path}\\similarityScores_sc3.json"
    with open(file_path, "a") as json_file:
        json.dump(full_dict, json_file)
