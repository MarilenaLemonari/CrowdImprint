from CNN36class import *
from helper_functions import *
from data_loader import * 
from config import *

os.environ['WANDB_API_KEY']="29162836c3095b286c169bf889de71652ed3201b"

# TODO:
# cd C:\PROJECTS\SocialLandmarks
# Execute .\.venv\Scripts\activate 
# go to cd C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Data
# Execute python3 C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\SingleSwitch\train_model3.py

def validate(model, val_loader, criterion, device, CM = False):
    model.eval()
    val_loss = 0.0
    val_total = 0
    val_correct = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            labels = torch.Tensor(labels).long().to(device)
            inputs = inputs.unsqueeze(1).to(device)

            probs = model(inputs)
            loss = criterion(probs, labels)

            val_loss += loss.item()
            _, preds = torch.max(probs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

            if i == 0:
                y_val = labels
                _, y_val_pred = torch.max(probs, 1)
            else:
                y_val = torch.cat((y_val, labels), dim = 0)
                _, y_pred = torch.max(probs, 1)
                y_val_pred = torch.cat((y_val_pred, y_pred), dim = 0)



    val_loss_avg = val_loss / len(val_loader)
    val_acc_overall_avg = val_correct / val_total

    if CM == True:
        confusion = confusion_matrix(y_val, y_val_pred)
        print("Validation Confusion Matrix:")
        print(confusion)

        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')

    return val_loss_avg, val_acc_overall_avg
def train(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    model.train()
    epoch_losses = []
    acc_overall = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_total = 0
        correct = 0

        for i, (inputs, labels) in enumerate(train_loader):
            labels = torch.Tensor(labels).long().to(device)
            inputs = inputs.unsqueeze(1).to(device)

            optimizer.zero_grad()

            probs = model(inputs)
            loss = criterion(probs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, preds = torch.max(probs, 1)
            correct += (preds == labels).sum().item()
            epoch_total += labels.size(0)
            

            if (i+1) % 100 == 0:
                # Run validation
                val_loss, val_acc_overall = validate(model, val_loader, criterion, device)
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}]:: Loss: [{epoch_loss / (i + 1):.4f}], Accuracy: {(correct * 100) /epoch_total:.2f}%, Validation:: Loss: [{val_loss:.4f}], Accuracy: {val_acc_overall*100:.2f}%')


        epoch_losses.append(epoch_loss)
        acc_overall.append(correct / epoch_total)

        val_loss, val_acc_overall = validate(model, val_loader, criterion, device, CM = True)
        
        # Log metrics to wandb
        wandb.log({"epoch": epoch+1, "train_loss": epoch_loss / len(train_loader), "train_acc": correct /epoch_total,
                    "val_loss": val_loss, "val_acc": val_acc_overall})

        
        
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

    # Data and Model SETUP:
    images, gt = load_data()
    print("SUCCESS! Data Loaded. Details: ", images.shape, len(gt))
    trainloader, valoader, config = setup_config(True, images, gt)
    model, criterion, optimizer, device = instantiate_model()

    # Train the model
    wandb.watch(model, log_freq=1)
    epoch_losses, acc_overall = train(model, trainloader, valoader, criterion, optimizer, device, config.epochs)
    wandb.finish()

    # Saving:
    torch.save(model.state_dict(), "C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\SingleSwitch\\test_model.h5")
    print("SUCCESS! Model is daved.")
    training_metrics = {
    "epoch_losses": epoch_losses,
    "acc_overall": acc_overall
    }
    filename = 'training_metrics.json'
    with open(filename, 'w') as json_file:
        json.dump(training_metrics, json_file, indent=4)
    print(f"Data successfully written to {filename}")
