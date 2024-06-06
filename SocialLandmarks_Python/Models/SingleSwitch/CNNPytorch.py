from imports import *

# python3 C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\SingleSwitch\CNNPytorch.py

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout1 = nn.Dropout(0.3)
        
        # Second Convolutional Block
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout2 = nn.Dropout(0.3)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(32 * 8 * 8, 2048)
        self.bn3 = nn.BatchNorm1d(2048)
        self.dropout3 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(2048, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.dropout4 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(1024, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.dropout5 = nn.Dropout(0.5)
        
        # Output Layer
        self.fc4 = nn.Linear(128, 25)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        
        x = x.view(-1, 32 * 8 * 8)
        
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout3(x)
        
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.dropout4(x)
        
        x = F.relu(self.bn5(self.fc3(x)))
        x = self.dropout5(x)
        
        x = self.fc4(x)
        
        return F.log_softmax(x, dim=1)
    
def instantiate_model():
    model = CNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("SUCCESS! Model Instantiated.")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

    return model, criterion, optimizer, device
    
if __name__ ==  '__main__':
    model, criterion, optimizer, device = instantiate_model()

    batch_size = 32
    inputs= torch.randn(batch_size, 1, 32, 32, requires_grad=True)
    print(inputs.shape)
    preds=  model(inputs)
    print(preds.shape)
    exit()