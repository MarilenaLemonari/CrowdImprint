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


def validate(model, val_loader, criterion, device, CM = False, name = "Validation"):
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

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_total = 0
        correct = 0

        for i, (inputs, labels) in enumerate(train_loader):
            labels = torch.Tensor(labels).long().to(device)
            inputs = inputs.unsqueeze(1).to(device)
            print(inputs.shape)
            print(labels.shape)
            exit()

            optimizer.zero_grad()

            probs = model(inputs)
            loss = criterion(probs, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, preds = torch.max(probs, 1)
            correct += (preds == labels).sum().item()
            epoch_total += labels.size(0)
            

            if (i+1) % 200 == 0:
                # Run validation
                val_loss, val_acc_overall = validate(model, val_loader, criterion, device)
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}]:: Loss: [{epoch_loss / (i + 1):.4f}], Accuracy: {(correct * 100) /epoch_total:.2f}%, Validation:: Loss: [{val_loss:.4f}], Accuracy: {val_acc_overall*100:.2f}%')


        epoch_losses.append(epoch_loss)
        acc_overall.append(correct / epoch_total)

        val_loss, val_acc_overall = validate(model, val_loader, criterion, device, CM = True)
        
        # Log metrics to wandb
        wandb.log({"epoch": epoch+1, "loss": epoch_loss / len(train_loader), "accuracy": correct /epoch_total,
                    "val_loss": val_loss, "val_accuracy": val_acc_overall})

        
        
    print('Finished Training')
    return epoch_losses, acc_overall

if __name__ ==  '__main__':
    """
    Conv1d.
    """

    start_time = time.time()
    X, labels, seq_len  = load_traj_data()
    print("SUCCESS! Data Loaded. Details: ", X.shape, labels.shape)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Loading Time: {elapsed_time:.2f} seconds")

#     model = instantiate_model(seq_length = seq_len)
#     print("SUCCESS! Model is Instantiated.")
#     X_train, X_val, y_train, y_val, config = setup_config_conv1d(X, labels)
#     print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
    model, criterion, optimizer, device = instantiate_model(seq_length=seq_len)
    print("SUCCESS! Model is Instantiated.")
    trainloader, valoader, config = setup_config(True, X, labels)

    start_time = time.time()
    # Train the model
    wandb.watch(model)
    epoch_losses, acc_overall = train(model, trainloader, valoader, criterion, optimizer, device, config.epochs)
    wandb.finish()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training Time: {elapsed_time:.2f} seconds")  
#     model.fit(X_train, y_train, epochs=config.epochs, batch_size=config.batch_size, validation_data=(X_val, y_val),
#           callbacks=[WandbCallback()])
#     wandb.finish()

#     # Saving:
#     # # Example usage
#     # eval_results = model.evaluate(X_val, y_val)
#     # print(f"Validation Loss: {eval_results[0]}")
#     # print(f"Validation Accuracy: {eval_results[1]}")
#     t_cm = make_cm(model, X_train, y_train, "Training")
#     v_cm  = make_cm(model, X_val, y_val, "Validation")
#     print("Training CM")
#     print(t_cm)
#     print("Validation CM:")
#     print(v_cm)
#     model.save("C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\Conv1d\model_test.h5")
#     print("SUCCESS! Model is Saved.")