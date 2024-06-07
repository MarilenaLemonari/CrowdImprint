from imports import *
from data_loader import *
from CNNPytorch import *
from config import *

os.environ['WANDB_API_KEY']="29162836c3095b286c169bf889de71652ed3201b"

# TODO:
# cd C:\PROJECTS\SocialLandmarks
# Execute .\.venv\Scripts\activate 
# go to cd C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data
# Execute python3 C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\SingleSwitch\train_pytorch_model.py

def validate(model, val_loader, criterion, device, CM = False, name = "Validation"):
    model.eval()
    val_loss = 0.0
    val_total = 0
    val_correct = 0

    # addition
    correct1 = 0
    correct2 = 0
    epoch_loss1 = 0
    epoch_loss2 = 0

    pred1_dict = {0 : 1, 1:1, 2:1, 3:1, 4:1,
        5:2, 6:2, 7:2, 8:2, 9:2,
        10:3, 11:3,12:3, 13:3, 14:3,
        15:4, 16:4, 17:4, 18:4, 19:4,
        20:5, 21:5, 22:5, 23:5, 24:5}
    pred2_dict = {0:1, 1:2, 2:3, 3:4, 4:5,
        5:1, 6:2, 7:3, 8:4, 9:5,
        10:1, 11:2, 12:3, 13:4, 14:5,
        15:1, 16:2, 17:3, 18:4, 19:5,
        20:1, 21:2, 22:3, 23:4, 24:5}

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            labels = torch.Tensor(labels).long().to(device)
            inputs = inputs.unsqueeze(1).to(device)

            # probs = model(inputs)
            # loss = criterion(probs, labels)
            probs_1, probs_2 = model(inputs)
            loss_1 = criterion(probs_1,  torch.tensor([pred1_dict[int(key)] for key in labels]) - 1)
            loss_2 = criterion(probs_2,  torch.tensor([pred2_dict[int(key)] for key in labels]) - 1)
            loss = loss_1 + loss_2


            val_loss += loss.item()
            epoch_loss1 += loss_1.item()
            epoch_loss2 += loss_2.item()
            # _, preds = torch.max(probs, 1)
            # correct += (preds == labels).sum().item()
            # epoch_total += labels.size(0)
            correct_1, total, matches1 = accuracy_f(torch.tensor([pred1_dict[int(key)] for key in labels]), probs_1)
            correct_2, total, matches2 = accuracy_f(torch.tensor([pred2_dict[int(key)] for key in labels]), probs_2)
            correct1 += correct_1
            correct2 += correct_2
            val_correct += accuracy_overall(matches1, matches2)
            val_total += labels.size(0)

            # if i == 0:
            #     y_val = labels
            #     _, y_val_pred = torch.max(probs, 1)
            # else:
            #     y_val = torch.cat((y_val, labels), dim = 0)
            #     _, y_pred = torch.max(probs, 1)
            #     y_val_pred = torch.cat((y_val_pred, y_pred), dim = 0)



    val_loss_avg = val_loss / len(val_loader)
    val_acc_overall_avg = val_correct / val_total

    if CM == True:
        confusion = confusion_matrix(y_val, y_val_pred)
        print(f"{name} Confusion Matrix:")
        print(confusion)

        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion, annot=False, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{name} Confusion Matrix')
        plt.savefig(f'{name}_confusion_matrix.png')

    return val_loss_avg, val_acc_overall_avg
def train(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    model.train()
    epoch_losses = []
    acc_overall = []

    pred1_dict = {0 : 1, 1:1, 2:1, 3:1, 4:1,
        5:2, 6:2, 7:2, 8:2, 9:2,
        10:3, 11:3,12:3, 13:3, 14:3,
        15:4, 16:4, 17:4, 18:4, 19:4,
        20:5, 21:5, 22:5, 23:5, 24:5}
    pred2_dict = {0:1, 1:2, 2:3, 3:4, 4:5,
        5:1, 6:2, 7:3, 8:4, 9:5,
        10:1, 11:2, 12:3, 13:4, 14:5,
        15:1, 16:2, 17:3, 18:4, 19:5,
        20:1, 21:2, 22:3, 23:4, 24:5}

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_total = 0
        correct = 0

        # addition
        correct1 = 0
        correct2 = 0
        epoch_loss1 = 0
        epoch_loss2 = 0

        for i, (inputs, labels) in enumerate(train_loader):
            labels = torch.Tensor(labels).long().to(device)
            inputs = inputs.unsqueeze(1).to(device)

            optimizer.zero_grad()

            # probs = model(inputs)
            # loss = criterion(probs, labels)
            probs_1, probs_2 = model(inputs)
            loss_1 = criterion(probs_1, torch.tensor([pred1_dict[int(key)] for key in labels]) - 1)
            loss_2 = criterion(probs_2,  torch.tensor([pred2_dict[int(key)] for key in labels]) - 1)
            loss = loss_1 + loss_2

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_loss1 += loss_1.item()
            epoch_loss2 += loss_2.item()
            # _, preds = torch.max(probs, 1)
            # correct += (preds == labels).sum().item()
            # epoch_total += labels.size(0)
            correct_1, total, matches1 = accuracy_f(torch.tensor([pred1_dict[int(key)] for key in labels]), probs_1)
            correct_2, total, matches2 = accuracy_f(torch.tensor([pred2_dict[int(key)] for key in labels]), probs_2)
            correct1 += correct_1
            correct2 += correct_2
            correct += accuracy_overall(matches1, matches2)
            epoch_total += labels.size(0)
            

            if (i+1) % 200 == 0:
                # Run validation
                val_loss, val_acc_overall = validate(model, val_loader, criterion, device)
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}]:: Loss: [{epoch_loss / (i + 1):.4f}], Accuracy: {(correct * 100) /epoch_total:.2f}%, Validation:: Loss: [{val_loss:.4f}], Accuracy: {val_acc_overall*100:.2f}%')


        epoch_losses.append(epoch_loss)
        acc_overall.append(correct / epoch_total)

        val_loss, val_acc_overall = validate(model, val_loader, criterion, device, CM = False)
        
        # Log metrics to wandb
        # wandb.log({"epoch": epoch+1, "loss": epoch_loss / len(train_loader), "accuracy": correct /epoch_total,
        #             "val_loss": val_loss, "val_accuracy": val_acc_overall})
        wandb.log({"epoch": epoch+1, "loss": epoch_loss / len(train_loader), "accuracy": correct /epoch_total,
                    "val_loss": val_loss, "val_accuracy": val_acc_overall,
                    "loss1": epoch_loss1/len(train_loader), "loss2": epoch_loss2/len(train_loader), "acc1": correct1/epoch_total, "acc2": correct2/epoch_total})

        
        
    print('Finished Training')
    return epoch_losses, acc_overall


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

    start_time = time.time()
    # x_train, y_train  = load_data()
    # x_val, y_val  = load_data(val = True)
    images, gt  = load_data()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("SUCCESS! Data Loaded. Details: ", images.shape, len(gt))
    print(f"Loading Time: {elapsed_time:.2f} seconds")

    model, criterion, optimizer, device = instantiate_model()
    print("SUCCESS! Model is Instantiated.")
    trainloader, valoader, config = setup_config(True, images, gt)

    start_time = time.time()
    # Train the model
    wandb.watch(model)
    epoch_losses, acc_overall = train(model, trainloader, valoader, criterion, optimizer, device, config.epochs)
    wandb.finish()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training Time: {elapsed_time:.2f} seconds")

    # Saving:
    results_train = validate(model, trainloader, criterion, device, CM = True, name = "Training")
    results_val = validate(model, valoader, criterion, device, CM = True)

    performance_metrics = {
    "time": elapsed_time,
    "results_train": results_train,
    "results_val": results_val
    }
    filename = 'performance_metrics.json'
    with open(filename, 'w') as json_file:
        json.dump(performance_metrics, json_file, indent=4)
    torch.save(model.state_dict(), "C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\SingleSwitch\\model_test.pth")
    print("SUCCESS! Model is saved.")