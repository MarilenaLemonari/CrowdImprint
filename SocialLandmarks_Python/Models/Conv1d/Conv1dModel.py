from imports import *
from helper_functions import *

# python3 C:\PROJECTS\SocialLandmarks\SocialLandmarks_Python\Models\Conv1d\Conv1dModel.py

def keras_model(seq_length):
    num_features = 3 
    num_classes = 25

    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(seq_length, num_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

class Conv1DModel(nn.Module):
    def __init__(self, num_features, num_classes, seq_length):
        super(Conv1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2176, 128) 
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.flatten(x)
        # print(x.shape)
        # exit()
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def instantiate_model(seq_length):

    num_features = 3
    num_classes = 25

    model = Conv1DModel(num_features=num_features, num_classes=num_classes, seq_length=seq_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("SUCCESS! Model Instantiated.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.003)
    
    # model = keras_model(seq_length)
    # return model

    return model, criterion, optimizer, device


if __name__ ==  '__main__':
    model, criterion, optimizer, device = instantiate_model(seq_length=100)

    # Gnerate dummy data:
    seq_length = 150  
    num_features = 3  
    num_classes = 25 
    num_samples = 40000
    # X = np.random.rand(num_samples, seq_length, num_features)
    x = torch.randn(num_samples, num_features, seq_length)
    # y = np.random.randint(0, num_classes, size=(num_samples,))
    output = model(x)
    print(output.shape)
    print(output[0,:])
    