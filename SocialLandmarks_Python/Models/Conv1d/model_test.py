from imports import *
from Conv1dModel import *
from data_loader import *
from config import *

# Load test data:
folder_path = "C:\PROJECTS\SocialLandmarks\Data\Tracking\YOLO\Tracker\Trajectories"
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

csv_data = read_csv_files(folder_path)
n_csvs = len(csv_data)
dict_list = list(csv_data.items())

gt_dict = {"1_1": 0, "1_2": 1, "1_3": 2, "1_4": 3, "1_5": 4,
        "2_1": 5, "2_2": 6, "2_3": 7, "2_4": 8, "2_5": 9,
        "3_1": 10, "3_2": 11,"3_3": 12, "3_4": 13, "3_5": 14,
        "4_1": 15, "4_2": 16, "4_3": 17, "4_4": 18, "4_5": 19,
        "5_1": 20, "5_2": 21, "5_3": 22, "5_4": 23, "5_5": 24,
        "0_0": 12, "0_1": 10, "0_2": 11, "0_3": 12, "0_4": 13, "0_5": 14, "0_6": 13,
        "1_0": 2, "2_0": 7, "3_0": 12, "4_0": 17, "5_0": 22, "6_0":17,
        "1_6":3, "2_6":8, "3_6":13, "4_6":18, "5_6":23, "6_6":18,
        "6_1":15, "6_2":16, "6_3":17, "6_4":18, "6_5":19}

trajectories = []
seq_len = []
gt = []

for i in tqdm(range(n_csvs)):
    key, value = dict_list[i]
    # X
    traj_x = value["pos_x"]
    traj_z = value["pos_z"]
    sample_traj = np.column_stack((traj_x, traj_z))
    source_value = value["norm_source"]
    source_feature = np.ones((traj_x.shape[0],)) * source_value
    sample_traj = np.column_stack((sample_traj, source_feature))
    trajectories.append(sample_traj)
    seq_len.append(traj_x.shape[0])

    # Y
    class_index = key.split("class_")[1].split("_subject")[0] # key.split("IF_")[1].split("_T")[0]
    class_type = int(class_index) #gt_dict[class_index]
    gt.append(class_type)
    
max_seq_len = max(seq_len) 
X_padded = pad_sequences(trajectories, maxlen=max_seq_len, padding='post', dtype='float32')
gt = np.array(gt)
print(max_seq_len)
exit()

test_dataset = CustomDataset(X_padded, gt)
testloader  = DataLoader(test_dataset, batch_size=32, shuffle=False)


model, criterion, optimizer, device = instantiate_model(seq_length=150)
model.load_state_dict(torch.load('C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\Conv1d\\model_conv1d.pth'))
model.eval()

loss = 0.0
total = 0
correct = 0

with torch.no_grad():
    for i, (inputs, labels) in enumerate(testloader):
        labels = torch.Tensor(labels).long().to(device)
        # inputs = inputs.unsqueeze(1).to(device)
        inputs = inputs.permute(0, 2, 1).to(device)
        print(inputs.shape, labels.shape)

        probs = model(inputs)
        loss = criterion(probs, labels)

        loss += loss.item()
        _, preds = torch.max(probs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if i == 0:
            y_test = labels
            _, y_test_pred = torch.max(probs, 1)
        else:
            y_test = torch.cat((y_test, labels), dim = 0)
            _, y_pred = torch.max(probs, 1)
            y_test_pred = torch.cat((y_test_pred, y_pred), dim = 0)



loss_avg = loss / len(testloader)
acc_overall_avg = correct / total
print(loss_avg, acc_overall_avg)