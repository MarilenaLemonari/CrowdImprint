from imports import *

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Assuming input size is 32x32
        self.fc2 = nn.Linear(512, 36)  # 36 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Output: (batch_size, 32, 16, 16)
        x = self.pool(F.relu(self.conv2(x)))  # Output: (batch_size, 64, 8, 8)
        x = self.pool(F.relu(self.conv3(x)))  # Output: (batch_size, 128, 4, 4)
        x = x.view(-1, 128 * 4 * 4)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def instantiate_model():
    model = CNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("SUCCESS! Model Instantiated.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01) # L2 regularization.

    return model, criterion, optimizer, device
    
if __name__ ==  '__main__':
    model, criterion, optimizer, device = instantiate_model()

    # inputs= torch.randn(batch_size, 1, 32, 32, requires_grad=True)
    # print(inputs.shape)
    # preds=  model(inputs)
    # print(preds.shape)
    # exit()